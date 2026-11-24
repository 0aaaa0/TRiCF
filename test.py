import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader_rgb_ir
from utils.general import logger, coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xyxy2xywh2, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from evaluation_script.evaluation_script import evaluate
from utils.confluence import confluence_process


def test(data,
         weights=None,
         batch_size=32,
         imgsz=768,
         conf_thres=0.001,
         iou_thres=0.5,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=True,  # save auto-label confidences
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None,
         labels_list=None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        if save_txt:
            labels_dir = increment_path(Path(save_dir) / 'labels' / 'pred', exist_ok=False, mkdir=True)
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        labels_dir = save_dir / 'labels'

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        print(opt.task)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        val_path_rgb = data['val_rgb']
        val_path_ir = data['val_ir']
        dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt, pad=0.5, rect=opt.rect, prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    if nc == 1:
        s = ('%20s' + '%12s' * 10) % ('Class', 'Images', 'Labels', 'TP', 'FP', 'FN', 'F1', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')  # 设置进度条的显示信息
    else:
        s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.
    tp, fp, fn = 0, 0, 0
    loss = torch.zeros(4, device=device) # 修改为 4，与 ComputeLoss 一致
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        img_rgb = img[:, :3, :, :]
        img_ir = img[:, 3:, :, :]
        
        # 确保RGB和IR图像的数据类型与模型一致
        if half:
            img_rgb = img_rgb.half()
            img_ir = img_ir.half()

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, _, train_out = model(img_rgb, img_ir, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                # 传入 model 计算路由惩罚 Loss, 并截取前5个损失 (box, obj, cls, rank, route)
                loss += compute_loss([x.float() for x in train_out], targets)[1][:5]  

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            # out = confluence_process(out, 0.1, 0.5)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                i = labels_list.index(str(path.stem) + '.txt')
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    line = (i+1, *xywh, conf) if save_conf else (i+1, *xywh)  # label format
                    with open(labels_dir / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g,' * len(line)).rstrip(",") % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    #wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            #wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # mem = '%.4gM' % (torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0)
        # print(mem)

        # file_name, extension = os.path.splitext(path.name)
        # os.rename('/home/shen/Chenyf/FLIR-align-3class/feature_save/fea_20x20.png', '/home/shen/Chenyf/FLIR-align-3class/feature_save/'+file_name+'_20x20'+'.png')
        # os.rename('/home/shen/Chenyf/FLIR-align-3class/feature_save/fea_40x40.png', '/home/shen/Chenyf/FLIR-align-3class/feature_save/'+file_name+'_40x40'+'.png')
        # os.rename('/home/shen/Chenyf/FLIR-align-3class/feature_save/fea_80x80.png', '/home/shen/Chenyf/FLIR-align-3class/feature_save/'+file_name+'_80x80'+'.png')

        # Plot images
        if plots and batch_i < 3:
            f1 = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img_rgb, targets, paths, f1, names), daemon=True).start()
            f2 = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img_rgb, output_to_target(out), paths, f2, names), daemon=True).start()

    # 保存所有预测框结果，后续用于MR的计算
    if save_txt:
        result_txt = labels_dir / 'result.txt'
        if result_txt.exists():
            result_txt.unlink()  # 删除旧的 result.txt
            
        temp = []
        files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'result.txt']
        files.sort()
        for file in files:
            with open(labels_dir / file, 'r') as f:
                temp.extend(f.readlines())
        
        with open(result_txt, 'w') as ff:  # 使用 'w' 模式重新写入
            ff.writelines(temp)

    # 计算MR指标 (仅在测试集上且 save_txt 为 True 时才计算)
    # 因为在每个 epoch 结束时的常规 val 阶段，没有必要（且很耗时）去跑完整的 MR 计算
    MR_reasonable, MR_day_reasonable, MR_night_reasonable, recall_reasonable = 0.0, 0.0, 0.0, 0.0
    MR_all, MR_day, MR_night, MR_near, MR_medium, MR_far, MR_none, MR_partial, MR_heavy, recall_all = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    if save_txt:
        try:
            # 根据数据集选择对应的 annotation 文件
            if 'KAIST' in str(data):
                annFile = 'evaluation_script/KAIST_annotation.json'
            elif 'VEDAI' in str(data):
                annFile = 'evaluation_script/VEDAI_annotation.json' # 假设存在，如果不存在会进入 except
            else:
                annFile = 'evaluation_script/KAIST_annotation.json' # 默认

            rstFiles = os.path.abspath(str(labels_dir / 'result.txt'))
            # 针对部分环境可能出现的路径前缀问题进行清理
            if rstFiles.startswith('./'):
                rstFiles = rstFiles[2:]
            phase = "Multispectral"
            if os.path.exists(annFile):
                MR = evaluate(annFile, rstFiles, phase)
                
                # 计算Reasonable条件下的MR
                MR['all'].evaluate(0)
                MR['all'].accumulate()
                MR_reasonable = MR['all'].summarize(0)
                MR['day'].evaluate(0)
                MR['day'].accumulate()
                MR_day_reasonable = MR['day'].summarize(0)
                MR['night'].evaluate(0)
                MR['night'].accumulate()
                MR_night_reasonable = MR['night'].summarize(0)
                try:
                    recall_reasonable = 1 - MR['all'].eval['yy'][0][-1]
                except (IndexError, TypeError):
                    recall_reasonable = 0.0
            else:
                print(f"Warning: Annotation file {annFile} not found. Skipping MR calculation.")
        except Exception as e:
            print(f"Warning: Failed to compute MR metrics. {e}")
    MRresult = [
        # All set
        MR_all, MR_day, MR_night, MR_near, MR_medium, MR_far, MR_none, MR_partial, MR_heavy, recall_all,
        # Reasonable set
        MR_reasonable, MR_day_reasonable, MR_night_reasonable, recall_reasonable
    ]

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    a = len(stats)
    b = stats[0].any()
    if len(stats) and stats[0].any():
        tp, fp, fn, p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    if nc > 1:
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 5  # print format
        logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))
    else:
        pf = '%20s' + '%12i' * 2 + '%12.4g' * 8  # print format
        logger.info(pf % ('all', seen, nt.sum(), tp, fp, fn, f1, mp, mr, map50, map))
    logger.info(('%20s' + '%11s' * 3) % ('MR-Reasonable', 'MR-day-R', 'MR-night-R', 'Recall-R'))
    logger.info(('%20.2f' + '%11.2f' * 3) % (MR_reasonable * 100, MR_day_reasonable * 100, MR_night_reasonable * 100, recall_reasonable * 100))
    logger.info(('%20s' + '%11s' * 9) % ('MR-All', 'MR-day', 'MR-night', 'MR-near', 'MR-medium', 'MR-far', 'MR-none', 'MR-partial', 'MR-heavy', 'Recall-All'))
    logger.info(('%20.2f' + '%11.2f' * 9) % (MR_all * 100, MR_day * 100, MR_night * 100, MR_near * 100, MR_medium * 100, MR_far * 100, MR_none * 100, MR_partial * 100, MR_heavy * 100, recall_all * 100))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            logger.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        logger.info(f"Results saved to {save_dir}{s}")
    
    # 清理非叶子节点张量，避免后续 deepcopy 时报错
    for m in model.modules():
        if hasattr(m, 'last_routing_weights'):
            m.last_routing_weights = None

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    if not isinstance(tp, int):
        return (tp[0], fp[0], fn[0], f1[0], mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, MRresult, t
    else:
        return (tp, fp, fn, f1, mp, mr, map50, map,
                *(loss.cpu() / len(dataloader)).tolist()), maps, MRresult, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/VEDAI/seu_seu_pam12/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='./data/multispectral/VEDAI.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=768, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular inference')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--fold', type=str, default=None, help='specific fold to test (e.g. fold01) or "all" for cross-validation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    print(opt.data)
    check_requirements()

    if opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot

    elif opt.task in ('train', 'val', 'test'):  # run normally
        # Load the base YAML to get configuration
        with open(opt.data) as f:
            base_data = yaml.safe_load(f)

        # Handle folds
        folds_to_test = [None]
        if opt.fold:
            if opt.fold == 'all':
                folds_to_test = base_data.get('folds', [f'fold{i:02d}' for i in range(1, 11)])
            else:
                folds_to_test = [opt.fold]
        
        all_results = []
        
        for fold in folds_to_test:
            if fold:
                print(f"\n--- Testing Fold: {fold} ---")
                # Dynamically update validation paths for the specific fold
                # Assuming structure: path/foldXX/visible/test and path/foldXX/infrared/test
                base_data['val_rgb'] = [os.path.join(base_data['path'], fold, 'visible', 'test').replace("\\", "/")]
                base_data['val_ir'] = [os.path.join(base_data['path'], fold, 'infrared', 'test').replace("\\", "/")]
                
                # Update save name to include fold
                original_name = opt.name
                opt.name = f"{original_name}_{fold}"
            
            # Check if running VEDAI or other multispectral datasets
            if opt.data in ['./data/multispectral/FLIR-align-3class.yaml', './data/multispectral/FLIR-ADAS.yaml', './data/multispectral/VEDAI.yaml' ,'./data/multispectral/VEDAI_folds.yaml', './data/multispectral/LLVIP.yaml', './data/multispectral/M3FD.yaml']:
                opt.verbose = True
                
                # Derive labels path from val_rgb
                val_path = base_data['val_rgb']
                if isinstance(val_path, list):
                    val_path = val_path[0]
                
                # Replace 'visible' with 'labels' more robustly
                # Example: fold10/visible/test -> fold10/labels/test
                parts = val_path.replace('\\', '/').split('/')
                if 'visible' in parts:
                    idx = parts.index('visible')
                    parts[idx] = 'labels'
                
                p_rel = '/'.join(parts)
                p = p_rel if os.path.isabs(p_rel) else os.path.join(base_data.get('path', ''), p_rel)
                
                # Handle potential 'images' subfolder
                if 'images' in p:
                    p = p.replace('/images', '')
                    
                if os.path.exists(p):
                    labels_list = os.listdir(p)
                    labels_list.sort()
                else:
                    labels_list = []
                    print(f"Warning: Labels path {p} does not exist. labels_list is empty.")
            else:
                labels_list = []

            results, maps, MRresult, t = test(base_data, # Pass the updated dict instead of path
                 opt.weights,
                 opt.batch_size,
                 opt.img_size,
                 opt.conf_thres,
                 opt.iou_thres,
                 opt.save_json,
                 opt.single_cls,
                 opt.augment,
                 opt.verbose,
                 save_txt=opt.save_txt | opt.save_hybrid,
                 save_hybrid=opt.save_hybrid,
                 save_conf=opt.save_conf,
                 opt=opt,
                 labels_list=labels_list
                 )
            
            if fold:
                all_results.append(results)
                opt.name = original_name # Restore original name
        
        # If testing multiple folds, aggregate results
        if len(all_results) > 1:
            print("\n" + "="*50)
            print(f"  Cross-Validation Results ({len(all_results)} folds)")
            print("="*50)
            metrics = ['P', 'R', 'mAP@.5', 'mAP@.5:.95']
            # results index: P=4, R=5, mAP50=6, mAP=7
            res_array = np.array(all_results)
            mean_res = res_array.mean(axis=0)
            std_res = res_array.std(axis=0)
            
            for i, m in enumerate(metrics):
                idx = i + 4
                print(f"{m:>10}: {mean_res[idx]:.4f} (+/- {std_res[idx]:.4f})")
            print("="*50)
