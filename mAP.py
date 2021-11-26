import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-pred", default="./prediction.txt")
parser.add_argument("-gt",default="./ground_truth.txt")

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

def iou(bbox1, bbox2):
    x1,y1,w1,h1 = bbox1
    x2,y2,w2,h2 = bbox2

    x = max(x1, x2)
    y = max(y1, y2)
    xmax = min(x1+w1, y1+h1)
    ymax = min(x2+w2, y2+h2)
    intersection_area = (xmax-x) * (ymax-y)
    if intersection_area < 0:
        print(f"bbox error {bbox1} or {bbox2}")
        return 0

    box1_area = w1*h1
    box2_area = w2*h2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area




def main():
    args = parser.parse_args()

    # load files
    with open(args.pred,"r") as f:
        pred = f.read().strip().split("\n")
    with open(args.gt,"r") as f:
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

    total_dict = {}
    for file,paths in file_dict.items():
        gt_path,pred_path = paths

        gt_anno_dict = {}
        with open(gt_path, "r") as f:
            anno_list = f.read().strip().split("\n")
            for anno in anno_list:
                cls,x,y,w,h = anno.split()
                gt_anno_dict[cls] = gt_anno_dict.get(cls,[])+[list(map(float,[x,y,w,h]))]

        pred_anno_dict = {}
        if pred_dict:    
            with open(pred_path, "r") as f:
                anno_list = f.read().strip().split("\n")
                for anno in anno_list:
                    cls,x,y,w,h = anno.split()
                    pred_anno_dict[cls] = pred_anno_dict.get(cls,[])+[list(map(float,[x,y,w,h]))]

        total_dict[file] = {'gt':gt_anno_dict,'pred':pred_anno_dict}
        

        # calculate ap
        


        
            




if __name__=="__main__":
    main()