import glob
import shutil
import time

import pandas as pd
from nltk.metrics.distance import edit_distance
from absl import flags, app
from absl.flags import FLAGS


flags.DEFINE_string("gt","","path to ground truth")
flags.DEFINE_string("pred","path to prediction")
flags.DEFINE_boolean("wem",False,"calculate wem")


def main(_argv):
    ground_truth_path = FLAGS.gt
    prediction_path = FLAGS.pred

    with open(ground_truth_path,"r") as f:
        ground_truth_dict = {}
        for line in f.read().strip().split("\n"):
            file_name, target= line.split()
            ground_truth_dict[file_name] = target
    

    
    with open(prediction_path, "r") as f:
        prediction_dict = {}
        for line in f.read().strip().split("\n"):
            file_name, target = line.split()
            prediction_dict[file_name] = target

    if sorted(prediction_dict.keys()) == sorted(ground_truth_dict.keys()):
        print("all files matched")
    else:
        return

    total_count = 0
    total_ned = 0
    total_wem = 0

    for name, gt in ground_truth_dict.items():
        pred = prediction_dict[name]
        now_ned = 1-edit_distance(pred,gt) / max(len(gt), len(pred))
        total_ned += now_ned
        total_count += 1
        print(f"[{total_count}/{len(ground_truth_dict)}] {name}\n ground-truth : {gt}, prediction : {pred}\n 1-ned : {now_ned*100:.2f}\n")

        if pred==gt:
            total_wem += 1

    print(f"Final 1-NED : {total_ned/total_count*100:.3f}")
    
    if FLAGS.wem:
        print(f"Final WEM : {(total_wem/total_count)*100:.2f}\n")
    