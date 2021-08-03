import json
import os
import numpy as np
from random import sample
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_route", required=True,
                        help='data directory')
    parser.add_argument("--destination",required=True,default='./',
                        help='directory to store config json')
    args = parser.parse_args()
    return args

def check_fileroute(fileroute):
    ids = os.listdir(fileroute)
    for id in ids:
        # check relabel
        if not os.path.exists(os.path.join(fileroute, id, "relabel_1.nii.gz")):
            print("%d doesn't have relabel file.".format(id))
            os.system("rm -r %s".format(os.path.join(fileroute,id)))
        # remove labeler
        files = os.listdir(os.path.join(fileroute, id))
        for file in files:
            if 'Segmentation' in file and 'xz' not in file:
                shutil.move(os.path.join(fileroute, id, file), os.path.join(fileroute, id, 'Segmentation_label.nii.gz'))


def coronary_davinci(dir, dest):
    # generate id json
    ids = os.listdir(dir)
    n = len(ids)
    train_data = sample(ids, n//10*9)
    test_data = list(set(ids)-set(train_data))
    data = {'training_ids':train_data, "test_ids":test_data}
    jsondata = json.dumps(data)
    f = open(dest+'coronary_ids.json','w')
    f.write(jsondata)
    f.close()

def main():
    arg = parse_args()
    check_fileroute(arg.file_route)
    coronary_davinci(arg.file_route, arg.destination)


if __name__=='__main__':
    main()
