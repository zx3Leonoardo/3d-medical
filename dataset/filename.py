from genericpath import exists
import json
import os
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def load_ids(args, phase):
    if phase=='inference_id':
        ids_file = os.path.join(root, "config", args.inference_ids_filename)
    else:
        ids_file = os.path.join(root, "config", args.ids_filename)
    with open(ids_file,'r') as opened_file:
        ids = json.load(opened_file)
    for key, value in ids.items():
            args.sign[key] = value

def generate_filenames(args, phase):
    # get train/val/test filenames list
    load_ids(args, phase)

    filenames = list()
    for id in args.sign[phase]:
        img_filename = args.img_directory.format(subject=id)
        label_filename = args.label_directory.format(subject=id)
        if exists(img_filename) and exists(label_filename):
            filenames.append([img_filename, label_filename, id])
        else:
            for filename in (img_filename, label_filename):
                if not exists(filename):
                    raise FileNotFoundError
    return filenames