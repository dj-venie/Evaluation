import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-pred")
parser.add_argument("-gt")


def main():
    args = parser.parse_args()

    print(args.pred)
    print(args.gt)

if __name__=="__main__":
    main()