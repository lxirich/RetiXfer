import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from retixfer.train.data.dataset import Dataset
from retixfer.train.data.transforms import LoadImage, ImageScaling, CopyDict


def get_dataloader_splits(dataframe_path, data_root_path, targets_dict, shots_train="80%", shots_val="0%",
                          shots_test="20%", batch_size=8, num_workers=0, seed=0, task="classification",
                          size=(512, 512), batch_size_test=1, expert_knowledge=False, expert_knowledge_path=''):
    
    if task == "classification":
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling(size=size)])
    else:
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling()])

    
    data = []
    dataframe = pd.read_csv(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()                                      

        data_i = {"image_path": data_root_path + sample_df["image"]}                   
        if task == "classification":
            data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]           
        data.append(data_i)

    random.seed(seed)
    random.shuffle(data)

    if expert_knowledge:
        data_KD = []
        dataframe_KD = pd.read_csv(expert_knowledge_path)
        for i in range(len(dataframe_KD)):
            sample_df = dataframe_KD.loc[i, :].to_dict()
            data_i = {"image_path": data_root_path + sample_df["image"]}
            data_i["caption"] = sample_df["caption"]
            data_KD.append(data_i)

    labels = [data_i["label"] for data_i in data]                                      
    unique_labels = np.unique(labels)
    data_train, data_val, data_test = [], [], []

    for iLabel in unique_labels:
        idx = list(np.squeeze(np.argwhere(labels == iLabel)))

        train_samples = get_shots(shots_train, len(idx))
        val_samples = get_shots(shots_val, len(idx))
        test_samples = get_shots(shots_test, len(idx))

        [data_test.append(data[iidx]) for iidx in idx[:test_samples]]
        [data_train.append(data[iidx]) for iidx in idx[test_samples:test_samples+train_samples]]
        [data_val.append(data[iidx]) for iidx in idx[test_samples+train_samples:test_samples+train_samples+val_samples]]

    train_loader = get_loader(data_train, transforms, "train", batch_size, num_workers)
    val_loader = get_loader(data_val, transforms, "val", batch_size_test, num_workers)
    test_loader = get_loader(data_test, transforms, "test", batch_size_test, num_workers)

    KD_loader = None
    if expert_knowledge:
        KD_loader = get_loader(data_KD, transforms, "KD", batch_size, num_workers)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader, "KD":KD_loader}
    return loaders


def get_loader(data, transforms, split, batch_size, num_workers):

    if len(data) == 0:
        loader = None
    else:
        dataset = Dataset(data=data, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle = split == "train", num_workers=num_workers, drop_last=False)
    return loader


def get_shots(shots_str, N):
    if "%" in str(shots_str):
        shots_int = int(int(shots_str[:-1]) / 100 * N)
    else:
        shots_int = int(shots_str)
    return shots_int
