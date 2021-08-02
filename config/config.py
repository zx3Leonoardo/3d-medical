import argparse
parser = argparse.ArgumentParser()

# data in/out and dataset
parser.add_argument("--save", default="/home/qinzixin/models/coronary/")
parser.add_argument("--batch_size", default=1)
parser.add_argument("--ids_filename", default="/home/qinzixin/project/coronary/config/coronary_ids.json")
parser.add_argument("--img_directory", default="/home/qinzixin/data/coronary/{subject}/Segmentation-lt-label.nii.gz")
parser.add_argument("--label_directory", default="/home/qinzixin/data/coronary/{subject}/relabel_1.nii.gz")

# preprocess param
parser.add_argument("--n_labels", default=20)
#parser.add_argument("--norm_factor", default=200)

# hardware
parser.add_argument("--thread", default="4")
parser.add_argument("--gpu_ids", default=[0,1])

# train
parser.add_argument("--lr", default=0.0001, metavar='LR')
parser.add_argument("--epochs", default=200)
parser.add_argument("--early_stop", default=30)


arg = parser.parse_args()
