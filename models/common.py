# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path
import warnings
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath

from torch.nn import init, Sequential
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class VGGblock(nn.Module):
    def __init__(self, num_convs, c1, c2):
        super(VGGblock, self).__init__()
        self.blk = []
        for num in range(num_convs):
            if num == 0:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
            else:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vggblock = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.vggblock(x)

        return out


class ResNetblock(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1):
        super(ResNetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=self.expansion * c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * c2)

        self.shortcut = nn.Sequential()
        if stride != 1 or c1 != self.expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=self.expansion * c2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * c2),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 使用torch.add确保非就地操作
        out = torch.add(out, self.shortcut(x))
        out = F.relu(out)

        return out


class ResNetlayer(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1, is_first=False, num_blocks=1):
        super(ResNetlayer, self).__init__()
        self.blk = []
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.blk.append(ResNetblock(c1, c2, stride))
            for i in range(num_blocks - 1):
                self.blk.append(ResNetblock(self.expansion * c2, c2, 1))
            self.layer = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.layer(x)

        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 使用torch.add确保非就地操作
        if self.add:
            return torch.add(x, self.cv2(self.cv1(x)))
        else:
            return self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        # 关键修复：禁用就地激活
        self.act = nn.LeakyReLU(0.1, inplace=False)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Multiply(nn.Module):
    # Element-wise multiplication
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mul(x, y)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # 确保维度在有效范围内
        if isinstance(self.d, int) and (self.d > 10 or self.d < -10):  # 防止异常大的维度值
            self.d = 1  # 默认使用通道维度
        return torch.cat(x, self.d)


class Add(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        return x[0] * self.w + x[1] * (1 - self.w)


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):

        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1,
                                                                                                     3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3,
                                                                                                     1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1,
                                                                                                     3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1,
                                                                                                  3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3,
                                                                                                  1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1,
                                                                                                  3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                     self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))  # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                  self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))  # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return [rgb_fea_flat, ir_fea_flat]


class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1,
                 attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in
              range(n_layer)])

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        # new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        # new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        return new_fea


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (
            input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = ChannelAttention(in_channels=in_channels, reduction_ratio=reduction_ratio,
                                                  pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out


class Reshape(nn.Module):
    """Reshape tensor to specified shape"""

    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.shape})'


# ===================================================================SEU=======================================================
class SEU(nn.Module):
    """
    可见光与红外目标检测融合模块

    输入：来自主干网络（如YOLO的Backbone）的C3和/或C4层的多模态特征图
    双分支：模块接收两个并行的输入，分别是RGB模态特征和红外（IR）模态特征
    """

    def __init__(self, in_channels, out_channels=None):
        super(SEU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels // 2

        # 特征预处理模块
        self.rgb_preprocess = RGBPreprocessing(in_channels)
        self.ir_preprocess = IRPreprocessing(in_channels)

        # 融合核心模块
        self.fusion_core = FusionCore(in_channels)

        # 特征增强与输出模块
        self.enhancement = FeatureEnhancement(in_channels, self.out_channels)

    def forward(self, rgb_feature, ir_feature):
        """
        前向传播函数

        Args:
            rgb_feature: RGB模态特征图，形状为 [B, C, H, W]
            ir_feature: 红外模态特征图，形状为 [B, C, H, W]

        Returns:
            融合后的特征图，形状为 [B, out_channels, H, W]
        """
        # 保存原始特征用于残差连接
        original_feature = (rgb_feature + ir_feature) / 2

        # 特征预处理
        rgb_clean = self.rgb_preprocess(rgb_feature)
        ir_clean = self.ir_preprocess(ir_feature)

        # 融合核心
        fused_feature = self.fusion_core(rgb_clean, ir_clean)

        # 特征增强与输出
        enhanced_feature = self.enhancement(fused_feature, original_feature)

        return enhanced_feature


class RGBPreprocessing(nn.Module):
    """
    RGB特征预处理模块
    采用双边滤波 + 自适应阈值去噪
    """

    def __init__(self, channels):
        super(RGBPreprocessing, self).__init__()
        self.channels = channels

        # 双边滤波参数学习
        self.d_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.sigma_color = nn.Parameter(torch.ones(1) * 0.1)
        self.sigma_space = nn.Parameter(torch.ones(1) * 0.1)

        # 自适应阈值参数学习
        self.threshold_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.threshold_bn = nn.BatchNorm2d(channels)
        self.threshold_act = nn.Sigmoid()

    def bilateral_filter(self, x):
        """实现可微分的双边滤波 - 内存优化版本"""
        # 使用可学习的卷积模拟双边滤波的空间权重
        spatial_weight = self.d_conv(x)

        # 使用高斯滤波代替全局颜色相似度计算，以减少内存使用
        # 这是一个近似实现，避免创建巨大的h*w x h*w矩阵
        color_weight = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        color_weight = torch.exp(-(x - color_weight) ** 2 / (2 * self.sigma_color ** 2))

        # 组合空间权重和颜色权重
        weight = spatial_weight * color_weight

        # 应用权重 - 修复数据类型匹配问题
        kernel = torch.ones(x.size(1), 1, 3, 3, device=x.device, dtype=x.dtype) / 9
        filtered = F.conv2d(x * weight,
                            kernel,
                            padding=1,
                            groups=x.size(1))

        return filtered

    def adaptive_threshold(self, x):
        """实现自适应阈值去噪"""
        # 计算局部区域的平均值作为阈值
        local_mean = self.threshold_conv(x)
        local_mean = self.threshold_bn(local_mean)

        # 生成阈值掩码
        threshold_mask = self.threshold_act(x - local_mean)

        # 应用阈值掩码
        denoised = x * threshold_mask

        return denoised

    def forward(self, x):
        # 应用双边滤波
        filtered = self.bilateral_filter(x)

        # 应用自适应阈值去噪
        denoised = self.adaptive_threshold(filtered)

        return denoised


class IRPreprocessing(nn.Module):
    """
    红外特征预处理模块
    采用非局部均值滤波 + 热斑抑制
    """

    def __init__(self, channels):
        super(IRPreprocessing, self).__init__()
        self.channels = channels

        # 非局部均值滤波参数
        self.nl_conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.nl_conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.nl_conv3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.nl_gamma = nn.Parameter(torch.zeros(1))

        # 热斑抑制参数
        self.hotspot_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.hotspot_bn = nn.BatchNorm2d(channels)
        self.hotspot_act = nn.Sigmoid()

    def non_local_means(self, x):
        """实现非局部均值滤波"""
        batch_size, c, height, width = x.shape

        # 计算特征的嵌入表示
        f = self.nl_conv1(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B×(H*W)×C
        g = self.nl_conv2(x).view(batch_size, -1, height * width)  # B×C×(H*W)

        # 计算注意力图
        attention = torch.bmm(f, g)  # B×(H*W)×(H*W)
        attention = F.softmax(attention, dim=2)

        # 计算输出特征
        h_features = self.nl_conv3(x).view(batch_size, -1, height * width)  # B×C×(H*W)
        out = torch.bmm(h_features, attention.permute(0, 2, 1))
        out = out.view(batch_size, c, height, width)

        # 残差连接
        out = self.nl_gamma * out + x

        return out

    def hotspot_suppression(self, x):
        """实现热斑抑制"""
        # 检测潜在的热斑区域
        hotspot_map = self.hotspot_conv(x)
        hotspot_map = self.hotspot_bn(hotspot_map)
        hotspot_map = self.hotspot_act(hotspot_map)

        # 抑制热斑区域
        suppressed = x * (1 - hotspot_map)

        return suppressed

    def forward(self, x):
        # 应用非局部均值滤波
        filtered = self.non_local_means(x)

        # 应用热斑抑制
        suppressed = self.hotspot_suppression(filtered)

        return suppressed


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size // 2)  # 修改输入通道数为3
        self.sigmoid = nn.Sigmoid()

        # Sobel算子卷积核
        self.sobel_x = nn.Parameter(torch.Tensor([[-1, 0, 1],
                                                  [-2, 0, 2],
                                                  [-1, 0, 1]]).reshape(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.Tensor([[-1, -2, -1],
                                                  [0, 0, 0],
                                                  [1, 2, 1]]).reshape(1, 1, 3, 3), requires_grad=False)

    def forward(self, x):
        # 计算边缘响应
        b, c, h, w = x.shape
        x_avg = torch.mean(x, dim=1, keepdim=True)

        # 应用Sobel算子
        edge_x = F.conv2d(x_avg, self.sobel_x, padding=1)
        edge_y = F.conv2d(x_avg, self.sobel_y, padding=1)

        # 添加数值稳定性保护，避免NaN - 修复FP16溢出问题
        max_val = 30000.0  # 安全的FP16范围内的值
        edge_x = torch.clamp(edge_x, min=-max_val, max=max_val)
        edge_y = torch.clamp(edge_y, min=-max_val, max=max_val)
        edge_mag = torch.sqrt(torch.clamp(edge_x ** 2 + edge_y ** 2, min=1e-8))

        # 计算通道维度的最大值和平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接特征
        x = torch.cat([edge_mag, avg_out, max_out], dim=1)

        # 生成空间注意力图
        x = self.conv(x)
        x = self.sigmoid(x)

        return x


class ChannelAttention(nn.Module):
    """通道注意力模块 (SE模块)"""

    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class AdaptiveWeightModule(nn.Module):
    """动态权重系数生成模块"""

    def __init__(self, channels):
        super(AdaptiveWeightModule, self).__init__()
        self.bn_rgb = nn.BatchNorm2d(channels)
        self.bn_ir = nn.BatchNorm2d(channels)

        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, ir):
        # 获取BatchNorm层的统计特征
        rgb_var = torch.var(self.bn_rgb(rgb), dim=[2, 3])
        ir_var = torch.var(self.bn_ir(ir), dim=[2, 3])

        # 计算每个样本的方差均值
        rgb_var_mean = torch.mean(rgb_var, dim=1, keepdim=True)
        ir_var_mean = torch.mean(ir_var, dim=1, keepdim=True)

        # 拼接特征
        var_cat = torch.cat([rgb_var_mean, ir_var_mean], dim=1)

        # 生成权重系数 (范围在0.3-0.7之间)
        alpha = self.fc(var_cat)
        alpha = 0.3 + 0.4 * alpha  # 将范围从[0,1]映射到[0.3,0.7]

        return alpha.view(-1, 1, 1, 1)


class FusionCore(nn.Module):
    """融合核心模块"""

    def __init__(self, channels):
        super(FusionCore, self).__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channels)
        self.adaptive_weight = AdaptiveWeightModule(channels)

    def forward(self, rgb, ir):
        # 计算空间注意力
        spatial_map = self.spatial_attention(torch.cat([rgb, ir], dim=1))

        # 计算通道注意力
        rgb_channel_map = self.channel_attention(rgb)
        ir_channel_map = self.channel_attention(ir)

        # 计算动态权重系数
        alpha = self.adaptive_weight(rgb, ir)

        # 执行元素级融合
        rgb_weighted = rgb * spatial_map
        ir_weighted = ir * (1 - spatial_map)

        # 融合公式
        fused = alpha * rgb_weighted + (1 - alpha) * ir_weighted

        return fused


class FeatureEnhancement(nn.Module):
    """特征增强与输出模块"""

    def __init__(self, in_channels, out_channels):
        super(FeatureEnhancement, self).__init__()

        # 通道压缩
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fused_feature, original_feature):
        # 残差连接
        enhanced = fused_feature + original_feature

        # 通道压缩
        output = self.compress(enhanced)
        output = self.bn(output)
        output = self.relu(output)

        return output


# ===================================================================================PAM=============================================================================

# YOLOv5多模态融合模块
class PAM(nn.Module):
    """
    YOLOv5多模态融合模块

    将原始IR-C5特征转化为"聚焦真实目标、抑制伪干扰"的增强特征，
    并与RGB-C5特征进行融合，为后续检测提供高质量输入。
    """

    def __init__(self, channels, num_heads=8):
        super(PAM, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        # 阶段1：IR-C5物理先验增强
        self.ir_enhancement = IRC5PhysicalEnhancement(channels)

        # 阶段2：RGB-C5语义保持预处理
        self.rgb_purification = RGBC5SemanticPurification(channels)

        # 阶段3：物理先验引导的交叉注意力融合
        self.cross_attention = PhysicalGuidedCrossAttention(channels, num_heads)

        # 阶段4：融合特征后处理与输出
        self.post_processing = FusionPostProcessing(channels)

    def forward(self, rgb_c5, ir_c5):
        """
        前向传播函数

        Args:
            rgb_c5: RGB-C5特征图，形状为 [B, C, H, W]
            ir_c5: IR-C5特征图，形状为 [B, C, H, W]

        Returns:
            最终融合特征，形状为 [B, C, H, W]
        """
        # 阶段1：IR-C5物理先验增强
        ir_enhanced, physical_priors = self.ir_enhancement(ir_c5)

        # 阶段2：RGB-C5语义保持预处理
        rgb_purified = self.rgb_purification(rgb_c5)

        # 阶段3：物理先验引导的交叉注意力融合
        cross_fusion = self.cross_attention(ir_enhanced, rgb_purified, physical_priors)

        # 阶段4：融合特征后处理与输出
        final_fusion = self.post_processing(cross_fusion, ir_enhanced, rgb_purified, physical_priors)

        return final_fusion


class IRC5PhysicalEnhancement(nn.Module):
    """
    IR-C5物理先验增强模块

    将原始IR-C5特征转化为"聚焦真实目标、抑制伪干扰"的增强特征
    基于热辐射物理特性，计算4类核心物理先验
    """

    def __init__(self, channels):
        super(IRC5PhysicalEnhancement, self).__init__()
        self.channels = channels

        # 通道平均降维卷积
        self.channel_avg = nn.Conv2d(channels, 1, kernel_size=1)

        # 热辐射梯度先验 - Sobel算子
        self.sobel_x = nn.Parameter(torch.Tensor([[-1, 0, 1],
                                                  [-2, 0, 2],
                                                  [-1, 0, 1]]).reshape(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.Tensor([[-1, -2, -1],
                                                  [0, 0, 0],
                                                  [1, 2, 1]]).reshape(1, 1, 3, 3), requires_grad=False)

        # 多先验动态融合单元 - 轻量级MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(8, 16),  # 输入：4个先验的2个统计特征(占比和方差)
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),  # 输出：4个先验的权重
            nn.Softmax(dim=1)  # 确保权重和为1
        )

    def intensity_prior(self, intensity_map, kernel_size=3):
        """热辐射强度先验图计算"""
        # intensity_map: (B, 1, H, W) 通道平均后的IR-C5强度图
        pad = kernel_size // 2
        # 计算局部均值与标准差
        local_mean = F.avg_pool2d(intensity_map, kernel_size, stride=1, padding=pad)
        local_var = F.avg_pool2d((intensity_map - local_mean) ** 2, kernel_size, stride=1, padding=pad)
        local_std = torch.sqrt(local_var + 1e-6)
        # 自适应阈值
        threshold = local_mean + 0.5 * local_std
        # 生成先验图
        prior = (intensity_map - threshold) / (torch.max(intensity_map) - threshold + 1e-6)
        prior = torch.clamp(prior, min=0)  # 低于阈值的区域置0
        return torch.sigmoid(prior)  # 归一化到[0,1]

    def gradient_prior(self, intensity_map):
        """热辐射梯度先验图计算"""
        # 使用Sobel算子计算x/y方向梯度
        grad_x = F.conv2d(intensity_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(intensity_map, self.sobel_y, padding=1)
        # 计算梯度幅值
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        # 归一化
        max_magnitude = torch.max(grad_magnitude, dim=2, keepdim=True)[0]
        max_magnitude = torch.max(max_magnitude, dim=3, keepdim=True)[0]
        normalized_magnitude = grad_magnitude / (max_magnitude + 1e-6)
        return torch.sigmoid(normalized_magnitude)  # 归一化到[0,1]

    def continuity_prior(self, intensity_prior):
        """热区域空间连续性先验图计算"""
        # 在PyTorch中模拟连通域分析
        # 使用形态学操作代替OpenCV的connectedComponentsWithStats

        # 二值化 - 修复数据类型匹配问题
        binary = (intensity_prior > 0.5).to(intensity_prior.dtype)

        # 使用形态学操作增强连续区域
        kernel = torch.ones(3, 3, device=binary.device, dtype=binary.dtype)
        kernel = kernel.view(1, 1, 3, 3)

        # 开运算（先腐蚀后膨胀）去除小区域
        eroded = F.conv2d(binary, kernel, padding=1)
        eroded = (eroded >= 8).to(intensity_prior.dtype)  # 需要8个邻居都是1才保留

        dilated = F.conv2d(eroded, kernel, padding=1)
        dilated = (dilated > 0).to(intensity_prior.dtype)  # 只要有一个邻居是1就设为1

        # 高斯模糊平滑边缘
        smoothed = F.avg_pool2d(dilated, kernel_size=3, stride=1, padding=1)

        return smoothed

    def contrast_prior(self, intensity_map, kernel_size=5):
        """
        计算对比度先验，增强数值稳定性
        """
        B, C, H, W = intensity_map.shape
        
        # 限制输入值的范围，避免极值
        intensity_map = torch.clamp(intensity_map, min=-30000.0, max=30000.0)
        
        # 检查并处理NaN值
        intensity_map = torch.where(torch.isnan(intensity_map), torch.zeros_like(intensity_map), intensity_map)
        
        # 计算背景均值
        intensity_flat = intensity_map.view(B, -1)
        sorted_intensity, _ = torch.sort(intensity_flat, dim=1)
        bg_pixels = int(0.8 * intensity_flat.size(1))
        bg_mean = sorted_intensity[:, :bg_pixels].mean(dim=1, keepdim=True)
        bg_mean = bg_mean.view(B, 1, 1, 1).expand_as(intensity_map)
        
        # 限制bg_mean的范围
        bg_mean = torch.clamp(bg_mean, min=-30000.0, max=30000.0)
        bg_mean = torch.where(torch.isnan(bg_mean), torch.ones_like(bg_mean), bg_mean)
        
        # 计算分子和分母，增加更强的数值保护
        numerator = intensity_map - bg_mean
        numerator = torch.clamp(numerator, min=-30000.0, max=30000.0)
        numerator = torch.where(torch.isnan(numerator), torch.zeros_like(numerator), numerator)
        
        # 使用更大的epsilon和绝对值保护
        epsilon = 1e-2  # 进一步增大epsilon
        denominator = torch.clamp(torch.abs(bg_mean), min=epsilon) + epsilon
        
        # 安全的除法运算
        contrast = numerator / denominator
        
        # 检查除法结果是否包含NaN或无穷大
        contrast = torch.where(torch.isnan(contrast), torch.zeros_like(contrast), contrast)
        contrast = torch.where(torch.isinf(contrast), torch.zeros_like(contrast), contrast)
        
        # 限制对比度值的范围，避免极值
        contrast = torch.clamp(contrast, min=-5, max=5)
        
        # 放大高对比度区域响应
        result = torch.sigmoid(contrast * 2)  # 减小放大系数
        
        # 最终检查结果
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        
        return result

    def forward(self, x):
        # 通道平均降维
        x_avg = self.channel_avg(x)  # B×1×H×W

        # 1. 热辐射强度先验
        m_intensity = self.intensity_prior(x_avg)  # B×1×H×W

        # 2. 热辐射梯度先验
        m_gradient = self.gradient_prior(x_avg)  # B×1×H×W

        # 3. 热区域连续性先验
        m_continuity = self.continuity_prior(m_intensity)  # B×1×H×W

        # 4. 热对比度先验
        m_contrast = self.contrast_prior(x_avg)  # B×1×H×W

        # 收集所有先验图
        priors = [m_intensity, m_gradient, m_continuity, m_contrast]

        # 计算每张先验图的有效区域占比和特征方差
        batch_size = x.size(0)
        stats = []
        for prior in priors:
            # 有效区域占比 (大于0.5的区域)
            area_ratio = torch.mean((prior > 0.5).to(prior.dtype).view(batch_size, -1), dim=1)
            # 特征方差
            variance = torch.var(prior.view(batch_size, -1), dim=1)
            # 拼接统计特征
            stats.append(torch.cat([area_ratio.unsqueeze(1), variance.unsqueeze(1)], dim=1))

        # 拼接所有统计特征
        all_stats = torch.cat(stats, dim=1)  # B×8

        # 通过MLP计算动态权重
        weights = self.fusion_mlp(all_stats)  # B×4

        # 加权融合生成综合物理权重图
        m_total = torch.zeros_like(m_intensity)
        for i, prior in enumerate(priors):
            m_total = m_total + prior * weights[:, i].view(-1, 1, 1, 1)

        # 扩展权重图至C通道
        m_total_expand = m_total.expand_as(x)

        # 特征调制与输出单元
        # 将综合物理权重图作为空间注意力权重，对原始IR-C5特征进行逐像素调制
        modulated = x * m_total_expand
        residual = x * 0.3  # 残差连接，保留原始语义信息
        enhanced = modulated + residual

        # 返回增强特征和物理先验图（用于后续交叉注意力）
        return enhanced, [m_intensity, m_gradient, m_continuity, m_contrast, m_total]


class RGBC5SemanticPurification(nn.Module):
    """
    RGB-C5语义保持预处理模块

    保留"语义细节的检索源"，降低噪声干扰
    """

    def __init__(self, channels, out_channels=None):
        super(RGBC5SemanticPurification, self).__init__()
        self.channels = channels
        self.out_channels = out_channels if out_channels else channels

        # 语义噪声抑制 - SE模块
        self.se = ChannelAttention(channels)

        # 局部方差阈值
        self.unfold = nn.Unfold(kernel_size=3, padding=1)

        # 维度对齐（如果需要）
        if self.out_channels != channels:
            self.align = nn.Conv2d(channels, self.out_channels, kernel_size=1)
            self.align_bn = nn.BatchNorm2d(self.out_channels)
            self.align_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, c, h, w = x.shape

        # 通过SE模块强化高语义通道
        channel_weights = self.se(x)
        x_weighted = x * channel_weights

        # 计算3×3局部方差
        # 使用unfold操作获取每个位置的3×3邻域
        patches = self.unfold(x)  # B×(C*3*3)×(H*W)
        patches = patches.view(batch_size, c, 9, h, w)

        # 计算每个位置的局部方差
        local_mean = torch.mean(patches, dim=2, keepdim=True)
        local_var = torch.mean((patches - local_mean) ** 2, dim=2)  # B×C×H×W

        # 方差小于0.01的区域（平坦噪声）乘以0.5衰减 - 修复数据类型匹配问题
        noise_mask = (local_var < 0.01).to(local_var.dtype)
        suppression_factor = torch.ones_like(noise_mask) - noise_mask * 0.5

        # 应用抑制因子
        x_suppressed = x_weighted * suppression_factor

        # 维度对齐（如果需要）
        if self.out_channels != self.channels:
            x_suppressed = self.align(x_suppressed)
            x_suppressed = self.align_bn(x_suppressed)
            x_suppressed = self.align_relu(x_suppressed)

        return x_suppressed


class PhysicalGuidedCrossAttention(nn.Module):
    """
    物理先验引导的交叉注意力融合模块

    以增强后的IR-C5为Query（物理锚点），在语义净化后的RGB-C5中检索匹配的语义细节，
    通过多头机制实现多维度物理先验的精准引导。
    """

    def __init__(self, channels, num_heads=8):
        super(PhysicalGuidedCrossAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Query/Key/Value投影 - 使用独立的1×1卷积层
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # 输出投影
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, ir_enhanced, rgb_purified, physical_priors):
        batch_size, c, h, w = ir_enhanced.shape
        
        # 如果特征图太大，先下采样减少计算复杂度
        if h * w > 1600:  # 40x40
            scale_factor = (1600 / (h * w)) ** 0.5
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            ir_enhanced_small = F.interpolate(ir_enhanced, size=(new_h, new_w), mode='bilinear', align_corners=False)
            rgb_purified_small = F.interpolate(rgb_purified, size=(new_h, new_w), mode='bilinear', align_corners=False)
            # 递归调用处理小尺寸特征
            small_output = self.forward(ir_enhanced_small, rgb_purified_small, 
                                      [F.interpolate(p, size=(new_h, new_w), mode='bilinear', align_corners=False) 
                                       for p in physical_priors])
            # 上采样回原尺寸
            return F.interpolate(small_output, size=(h, w), mode='bilinear', align_corners=False)

        # 投影到Query/Key/Value空间
        q = self.q_proj(ir_enhanced)  # 增强后的IR-C5作为Query
        k = self.k_proj(rgb_purified)  # 语义净化后的RGB-C5作为Key
        v = self.v_proj(rgb_purified)  # 语义净化后的RGB-C5作为Value

        # 重塑为多头形式
        # 形状变换: [B, C, H, W] -> [B, h, H*W, d_k]
        q = q.view(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # B×h×(H*W)×d_k
        k = k.view(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # B×h×(H*W)×d_k
        v = v.view(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # B×h×(H*W)×d_k

        # 计算相似度矩阵: S = Q·K^T/√d_k
        # [B, h, H*W, d_k] × [B, h, d_k, H*W] -> [B, h, H*W, H*W]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 限制注意力分数范围，防止极端值
        attn_scores = torch.clamp(attn_scores, min=-10.0, max=10.0)

        # 物理先验调制 - 前4个头分别对应4类物理先验
        # 创建一个新的张量来存储调制后的注意力分数
        modulated_attn_scores = attn_scores.clone()

        for i in range(min(4, self.num_heads)):
            if i < len(physical_priors) - 1:  # 最后一个是综合物理权重图，不单独使用
                # 获取对应的物理先验图
                prior = physical_priors[i]
                # 数值稳定性处理
                prior = torch.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
                prior = torch.clamp(prior, 0.0, 1.0)
                # 归一化处理
                pmax = torch.amax(prior.view(batch_size, -1), dim=1, keepdim=True).view(batch_size, 1, 1, 1)
                prior = prior / (pmax + 1e-6)
                # 将先验图重塑为注意力权重形式
                prior_flat = prior.view(batch_size, 1, h * w, 1)
                # 使用更安全的调制方式：加权而非乘积
                prior_weight = prior_flat.squeeze(-1)  # B×1×(H*W)
                # 对注意力分数进行加权调制，而不是外积
                modulated_attn_scores[:, i] = attn_scores[:, i] * (0.5 + 0.5 * prior_weight)

        # 数值稳定性处理
        modulated_attn_scores = torch.nan_to_num(modulated_attn_scores, nan=0.0, posinf=10.0, neginf=-10.0)
        modulated_attn_scores = torch.clamp(modulated_attn_scores, min=-10.0, max=10.0)
        
        # 减去最大值防止溢出
        max_vals = modulated_attn_scores.max(dim=-1, keepdim=True).values
        modulated_attn_scores = modulated_attn_scores - max_vals
        
        # 注意力权重归一化: A = Softmax(S) - 在FP32中计算
        attn_weights = F.softmax(modulated_attn_scores.float(), dim=-1).to(modulated_attn_scores.dtype)  # B×h×(H*W)×(H*W)
        
        # 检查并处理NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)

        # 特征聚合: Fhead = A·V
        # [B, h, H*W, H*W] × [B, h, H*W, d_k] -> [B, h, H*W, d_k]
        context = torch.matmul(attn_weights, v)

        # 重塑回原始形状: [B, h, H*W, d_k] -> [B, C, H, W]
        context = context.permute(0, 1, 3, 2).contiguous().view(batch_size, c, h, w)

        # 输出投影 - 压缩至原通道数C
        output = self.out_proj(context)
        output = self.out_norm(output)
        output = self.out_relu(output)

        return output


class FusionPostProcessing(nn.Module):
    """
    融合特征后处理与输出模块

    采用"融合特征 + 双模态残差"的结构，避免单一模态信息丢失
    通过深度可分离卷积平滑融合特征的空间响应
    """

    def __init__(self, channels):
        super(FusionPostProcessing, self).__init__()
        self.channels = channels

        # 双模态残差权重
        self.res_weight_ir = 0.2  # IR-C5残差权重
        self.res_weight_rgb = 0.2  # RGB-C5残差权重

        # 特征平滑 - 使用3×3深度可分离卷积
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)  # 深度卷积
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)  # 逐点卷积
        self.smooth_bn = nn.BatchNorm2d(channels)
        self.smooth_relu = nn.ReLU(inplace=True)

    def forward(self, cross_fusion, ir_enhanced, rgb_purified, physical_priors):
        # 融合特征 + 双模态残差
        fusion_res = cross_fusion + ir_enhanced * self.res_weight_ir + rgb_purified * self.res_weight_rgb

        # 特征平滑（深度可分离卷积）- 避免权重突变导致的目标轮廓虚化
        smoothed = self.dw_conv(fusion_res)  # 深度卷积
        smoothed = self.pw_conv(smoothed)  # 逐点卷积
        smoothed = self.smooth_bn(smoothed)
        smoothed = self.smooth_relu(smoothed)

        return smoothed