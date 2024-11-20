# %%
# from dataset import ToSpectrograms, CusttomDataSnor
import pandas as pd  
import torch 
from sklearn.model_selection import train_test_split
from glob import glob 
import matplotlib.pyplot as plt
# from utils import plot_spectrogram
import librosa.display
from lit import LitClassification
import lightning as pl
import numpy as np   
from dataset import CusttomDataUrban
from torch.utils.data import DataLoader, WeightedRandomSampler
# %%

import argparse
parser = argparse.ArgumentParser(description='Mô tả về chương trình của bạn')
parser.add_argument('--name', type=str, default="", help='Mô tả cho arg1')

args = parser.parse_args()

name = args.name



df = pd.read_csv("data.csv")

df_train, df_test = train_test_split(df,
                        random_state=104,
                        test_size=0.2,
                        shuffle=True)

def get_class_distribution(dataset_obj):
    count_array = np.array([len(np.where(dataset_obj == t)[0]) for t in np.unique(dataset_obj)])
    return count_array

target_list = df_train['classID'].values

class_count = np.array([i for i in get_class_distribution(df_train['classID'].values)])
weight = 1. / class_count
samples_weight = np.array([weight[t] for t in target_list])
samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()

weighted_sampler = WeightedRandomSampler ( 
    weights = samples_weigth, 
    num_samples = len (samples_weigth), 
)

data_train = CusttomDataUrban(df_train)
train_dataloader =  DataLoader(data_train, batch_size=32, num_workers=2, sampler = weighted_sampler)

data_valid = CusttomDataUrban(df_test)
valid_dataloader =  DataLoader(data_valid, batch_size=32, num_workers=2)

tesst_dataloader =  DataLoader(data_valid, batch_size=32, num_workers=2)

next(iter(train_dataloader))

# checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max')

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="snoring", name=name)




model_lit = LitClassification()

trainer = pl.Trainer(max_epochs=100, logger=wandb_logger, precision='16-mixed')
trainer.fit(model_lit, train_dataloader, valid_dataloader)