import os
import glob
import numpy as np

from absl import flags, app
from absl.flags import FLAGS

#################################################
flags.DEFINE_string("gt","./mAP_example_files/ground_truth.txt","path to ground truth")
flags.DEFINE_string("pred","./mAP_example_files/prediction.txt","path to prediction")
flags.DEFINE_boolean("summary", False,"show only mAP")
#################################################

"""
-pred 
ex) prediction.txt
    ./img/imagename.jpg ./prediction/imagename.txt
    ...
-gt
ex) ground_truth.txt
    ./img/imagename.jpg ./ground_truth/imagename.txt
    ...

ex) imagename.txt
    classname x y w h
    ...
"""

IOU_THRESHOLD = 0.5

def cal_iou(bbox1, bbox2):
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2

    x = max(x1, x2)
    y = max(y1, y2)
    xmax = min(x1+w1, x2+w2)
    ymax = min(y1+h1, y2+h2)
    
    if (xmax>x) and (ymax >y):
        intersection_area = (xmax-x+1) * (ymax-y+1)
    else:
        return 0
    if intersection_area < 0:
        # print(f"bbox error {bbox1} or {bbox2}")
        return 0

    box1_area = (w1+1)*(h1+1)
    box2_area = (w2+1)*(h2+1)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def cal_ap(tp,cls_cnt):
    fp = [0 if i else 1 for i in tp]
            
    # compute precision/recall 
    cumsum = 0
    for ind, val in enumerate(fp):
        cumsum += val
        fp[ind] = cumsum
        
    cumsum = 0
    for ind, val in enumerate(tp):
        cumsum += val
        tp[ind] = cumsum

    rec = tp[:]
    for ind,val in enumerate(tp):
        rec[ind] = float(val) / cls_cnt

    prec = tp[:]
    for ind, val in enumerate(tp):
        prec[ind] = float(tp[ind]) / (fp[ind] + tp[ind])
    
    mrec = [0] + rec[:] + [1]
    mprec = [0] + prec[:] + [0]

    for i in range(len(mprec)-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])
    
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += ((mrec[i] - mrec[i-1])*mprec[i])
    
    return ap


def main(_argv):
    # load files
    with open(FLAGS.pred,"r") as f:
        pred = f.read().strip().split("\n")
    with open(FLAGS.gt,"r") as f:
        gt = f.read().strip().split("\n")

    # make file_dict
    pred_dict = {i.split()[0].split("/")[-1]:i.split()[1] for i in pred}
    gt_dict = {i.split()[0].split("/")[-1]:i.split()[1] for i in gt}

    remove_list = []
    for pred_name in pred_dict:
        if pred_name not in gt_dict:
            print(f"{pred_name} not in gt_dict")
            remove_list.append(pred_name)
    
    for r in remove_list:
        pred_dict.pop(r)

    if len(pred_dict)==0:
        return

    file_dict = {}
    for name,gt_path in gt_dict.items():
        file_dict[name] = [gt_path, pred_dict.get(name,"")]
    # load annotations

    total_tp_dict = {}
    total_dict = {'gt':{},'pred':{}}
    gt_cls_count = {}
    for file,paths in file_dict.items():
        gt_path,pred_path = paths

        gt_anno_dict = {}
        gt_counter = {}
        with open(gt_path, "r") as f:
            anno_list = f.read().strip().split("\n")
            for anno in anno_list:
                cls,x,y,w,h = anno.split()
                gt_anno_dict[cls] = gt_anno_dict.get(cls,[])+[list(map(float,[x,y,w,h]))]
                total_dict['gt'][cls] = total_dict['gt'].get(cls,[])+[list(map(float,[x,y,w,h]))]
                gt_counter[cls] = gt_counter.get(cls,0)+1
                gt_cls_count[cls] = gt_cls_count.get(cls,0)+1
                

        pred_anno_dict = {}
        if pred_dict:    
            with open(pred_path, "r") as f:
                anno_list = f.read().strip().split("\n")
                for anno in anno_list:
                    if anno=="":
                        continue
                    cls,score, x,y,w,h = anno.split()
                    pred_anno_dict[cls] = pred_anno_dict.get(cls,[])+[list(map(float,[score,x,y,w,h]))]
                    total_dict['pred'][cls] = total_dict['pred'].get(cls,[])+[list(map(float,[x,y,w,h]))]
                    total_tp_dict[cls] = total_tp_dict.get(cls,[])

        for cls in pred_anno_dict:
            pred_anno_dict[cls] = sorted(pred_anno_dict[cls],reverse=True)

        # calculate map per file
        ap_list = []
        for cls in gt_anno_dict:
            tp = []
            used_list = []
            for bbox1 in pred_anno_dict.get(cls,[]):
                score,bbox1 = bbox1[0], bbox1[1:]
                tmp_list = []
                for bbox2 in gt_anno_dict.get(cls,[]):
                    iou = cal_iou(bbox1, bbox2)
                    if iou > IOU_THRESHOLD:
                        tmp_list.append([iou,bbox2])

                if tmp_list:
                    _,now_matched = max(tmp_list)
                    if now_matched in used_list:
                        tp.append(0)
                        total_tp_dict[cls].append([score,0])
                    else:
                        used_list.append(now_matched)
                        tp.append(1)
                        total_tp_dict[cls].append([score,1])
                else:
                    tp.append(0)
                    total_tp_dict[cls].append([score,0])

            ap_list.append(cal_ap(tp, gt_counter[cls]))

        if not FLAGS.summary:
            if ap_list:
                print(f"{file}\nmAP : {sum(ap_list)/len(ap_list):.2f}")
            else:
                print(f"{file}\nmAP : 0")

        for cls in pred_anno_dict:
            if cls not in gt_anno_dict:
                for bbox2 in pred_anno_dict[cls]:
                    score,_,_,_,_ = bbox2
                    total_tp_dict[cls].append([score,0])
                
    
    # total mAP calculatea
    ap_dict = {}
    for cls,count in sorted(gt_cls_count.items()):
        tp = total_tp_dict.get(cls,[])
        tp = list(map(lambda x:x[1],sorted(tp, reverse=True)))
        ap_dict[cls] = cal_ap(tp,count)
        if not FLAGS.summary:
            print(f"{cls} : {ap_dict[cls]*100:.2f}")

    print(f"mAP : {sum(ap_dict.values())/len(ap_dict)*100:.2f}")

    # print(f"mAP : {sum(ap_list)/len(ap_list)*100:.2f}")
    # print(len(ap_list))

if __name__=="__main__":
    try:
        app.run(main)
    except SystemExit:
        pass