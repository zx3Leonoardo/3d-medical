from dataset.dataset import coronary_dataset
from config.config import *
import os
import SimpleITK as sitk
import numpy as np

route = '/home/qinzixin/projects/coronary/offline_check/img'

# origin img-label
args = arg
trainset = coronary_dataset(args, "training_ids")
img_path = trainset.filenames[0][0]
label_path = trainset.filenames[0][1]
os.system("cp {} {}/".format(img_path, route))

# transformed img-label
img_arr, label_arr = np.array(trainset[0][0]).astype(int), np.array(trainset[0][1]).astype(int)
img_arr, label_arr = sitk.GetImageFromArray(img_arr), sitk.GetImageFromArray(label_arr)
sitk.WriteImage(img_arr, '{}/img_transformed.nii.gz'.format(route))
sitk.WriteImage(label_arr, '{}/label_transformed.nii.gz'.format(route))

