import pandas as pd
import torch
import ast

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from retixfer.train.data.dataset import Dataset
from retixfer.train.data.transforms import LoadImage, SelectRelevantKeys, CopyDict,\
    ProduceDescription, AugmentDescription


def get_loader(dataframes_path, data_root_path, knowledge_dict_path, datasets, batch_size=8, num_workers=0,
               banned_categories=None, caption="A fundus photograph of [CLS]", augment_description=True,
               expert_knowledge=False):

    transforms = Compose([
        CopyDict(),                                             
        LoadImage(),                                            
        ProduceDescription(caption=caption),                    
        AugmentDescription(augment=augment_description),        
        SelectRelevantKeys()                                    
    ])
    
    if expert_knowledge:
        KD_transforms = Compose([CopyDict(), LoadImage()])     
       

    print("Setting assembly data...")
    data = []
    for iDataset in datasets:
        print("Processing data: " + iDataset)

        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv", keep_default_na=False)
        
        for col in ("categories", "atributes"):
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].apply(
                    lambda s: ast.literal_eval(s) if isinstance(s, str) and s.strip().startswith("[") else []
                )

        selected_id_list = range(len(dataframe))                     

        for i in selected_id_list:
            row = dataframe.loc[i, :]
            data_i = row.to_dict()

            data_i["categories"] = data_i.get("categories", [])
            data_i["atributes"]  = data_i.get("atributes", [])

            banned = False
            if banned_categories is not None:
                for iCat in data_i["categories"]:
                    if iCat in banned_categories:
                        banned = True
            if banned:
                continue

            data_i["image_name"] = data_i["image"]
            data_i["image_path"] = data_root_path + data_i["image"]
            data.append(data_i)
    print('Total assembly data samples: {}'.format(len(data)))
    
    if expert_knowledge:
        data_KD = []
        dataframe_KD = pd.read_csv(knowledge_dict_path)
        for i in range(len(dataframe_KD)):
            sample_df = dataframe_KD.loc[i, :].to_dict()
            data_i = {"image_path": data_root_path + sample_df["image"]}
            data_i["caption"] = sample_df["caption"]
            data_i["fundus_status"] = sample_df["fundus_status"]
            data_KD.append(data_i)
            
        print('Total assembly knowledge data samples: {}'.format(len(data_KD)))


    train_dataset = Dataset(data=data, transform=transforms)
    train_sampler = DistributedSampler(train_dataset)                 
    KD_loader = None
    if expert_knowledge:
        KD_dataset = Dataset(data=data_KD, transform=KD_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, pin_memory=True,persistent_workers=True, drop_last=True) 
    if expert_knowledge:
        weights = torch.ones(len(KD_dataset))  
        weightedRandomSampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=batch_size * len(train_loader))    
        KD_loader = DataLoader(KD_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weightedRandomSampler, pin_memory=True,persistent_workers=True, drop_last=True)

    dataloaders = {"train": train_loader, "KD":KD_loader}
    return dataloaders

