import json
import os
import numpy as np
from random import sample
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_route", required=True,
                        help='data directory')
    parser.add_argument("--destnation",required=True,
                        help='directory to store config json')
    args = parser.parse_args()
    return args

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


if __name__=='__main__':
    main()
