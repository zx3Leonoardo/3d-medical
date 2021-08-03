from typing import OrderedDict
from tensorboardX import SummaryWriter
import pandas as pd


class TrainLogger():
    def __init__(self, save_path, save_name) -> None:
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update(self, epoch, train_log, val_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)
        print("\033[0;33mTrain:\033[0m", train_log)
        print("\033[0;33mValid:\033[0m", val_log)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self, item):
        tmp = pd.DataFrame(item, index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv("%s/%s.csv" %(self.save_path, self.save_name), index=False)
    
    def update_tensorboard(self, item):
        if self.summary is None:
            self.summary = SummaryWriter("%s" % self.save_path)
        epoch = item['epoch']
        for key, value in item.items():
            if key!= 'epoch':self.summary.add_scalar(key, value, epoch)
