# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import torch.nn as nn

Linear = nn.Linear
ReLU   = nn.ReLU
Sigmoid = nn.Sigmoid
Tanh = nn.Tanh


sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

# from mmcv.ops import DeformConv2dPack as DCN

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m = nn.ModuleList(DCN(x, self.no * self.na, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1) for x in ch)  # output DCN conv3x3

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        logits_ = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                logits = x[i][..., 5:]

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, 'r', encoding='utf-8') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print(m)

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            # self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False):
        if augment:
            # 数据增强处理（需要同时处理两个模态）
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                xi2 = scale_img(x2.flip(fi) if fi else x2, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi, xi2)[0]  # forward
                # 后处理...
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, x2, profile)  # single-scale inference, train

    def forward_once(self, x, x2, profile=False):
        y, dt = [], []  # outputs
        
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                if m.f == -4:  # 特殊标记，表示使用IR输入
                    x = x2  # 直接使用IR输入
                elif isinstance(m.f, int):
                    x = y[m.f]  # 从单一层获取特征
                elif isinstance(m.f, list):
                    if isinstance(m, (SEU, PAM)):
                        # 对于SEU和PAM模块，不在这里处理特征
                        pass
                    else:
                        # 对于其他模块，合并特征
                        x = [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            # 处理不同类型的模块
            if isinstance(m, SEU):
                # 处理SEU模块 - 需要两个输入特征
                feat1 = y[m.f[0]]
                feat2 = y[m.f[1]]
                x = m(feat1, feat2)
            elif isinstance(m, PAM):
                # 处理PAM模块 - 需要两个输入特征
                rgb_feat = y[m.f[0]]
                ir_feat = y[m.f[1]]
                x = m(rgb_feat, ir_feat)
            elif isinstance(m.f, list) and not isinstance(m, (SEU, PAM)):
                # 处理其他多输入层
                if isinstance(x, list):
                    if len(x) == 1:
                        x = m(x[0])
                    else:
                        x = m(x)
                else:
                    x = m(x)
            else:
                # 标准模块
                x = m(x)

            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 处理 Linear 模块
        if m is nn.Linear:
            # 获取前一层的输出通道数
            if isinstance(f, int):
                c1 = ch[f]
            else:
                c1 = sum([ch[x] for x in f])

            # 计算输入特征数（假设前一层的输出形状为 [batch, channels, height, width]）
            # 对于全局平均池化后的特征，height 和 width 都是 1
            in_features = c1  # 如果前面是 AdaptiveAvgPool2d + Flatten

            # 添加输入特征数到参数列表
            args = [in_features] + args

            c2 = args[1]  # 输出特征数

        elif m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, Reshape, SEU, PAM]:  # 添加对SEU和PAM模块的支持

            if m is Focus:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is Conv and args[0] == 64:  # new
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is SEU:
                # 正确的 SEU 通道处理：
                # - in_channels 取每个支路的通道（假设两个输入通道一致，使用第一个分支的通道）
                # - 默认 out_channels = in_channels // 2（当只提供一个参数时）
                in_ch = ch[f[0]] if not isinstance(f, int) else ch[f]
                target_in = args[0]
                out_ch = (target_in // 2) if len(args) == 1 else args[1]
                if out_ch != no:
                    out_ch = make_divisible(out_ch * gw, 8)
                c2 = out_ch
                args = [in_ch, c2]  # 传入 SEU(in_channels, out_channels)

            elif m is PAM:
                # 正确的 PAM 通道处理：
                # - channels 使用每个支路的通道（假设两个输入通道一致）
                # - 输出通道与输入 channels 一致
                in_ch = ch[f[0]] if not isinstance(f, int) else ch[f]
                c2 = in_ch
                args = [c2]  # PAM(channels)

            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is ResNetlayer:
            if args[3] == True:
                c2 = args[1]
            else:
                c2 = args[1] * 4
        elif m is VGGblock:
            c2 = args[2]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])

            args = [c2]
        elif m is Add:
            c2 = ch[f[0]]
            args = []
        elif m is Add2:
            c2 = ch[f] if isinstance(f, int) else sum([ch[x] for x in f])

            args = [c2, args[1]]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]
        elif m is Reshape:
            c2 = args[0]  # 输出通道数是形状的第一个维度
        elif m is TransformerFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        # m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        try:
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        except Exception as e:
            print(f"\n❌ Error at module: {m.__name__}")
            print(f"   args = {args}")
            print(f"   n = {n}")
            raise e
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params

        # ✅ Debug 打印 c1/c2 信息
        if isinstance(f, int):
            f_str = str(f)
            c1_str = ch[f] if f != -1 else 'prev'
        else:
            f_str = ','.join(map(str, f))
            c1_str = [ch[x] for x in f]

        print(f"[parse_model] layer {i:2d} | from={f_str:6s} | {t:<25} | c1={c1_str} -> c2={c2} | args={args}")

        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # 原有日志
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []

        ch.append(c2)

    try:
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
    except Exception as e:
        print(f"❌ Error at module {m.__name__}, args={args}, n={n}")
        raise e

    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='F:/software/project/ICAFusion-main/ICAFusion-mai/models/yolov5_physical.yaml',
                        help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    print(device)

    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    input_ir = torch.Tensor(8, 1, 640, 640).to(device)  # 红外图像是单通道

    output = model(input_rgb, input_ir)