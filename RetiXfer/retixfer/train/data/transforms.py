import numpy as np
import random
import torch
import copy
import os

from PIL import Image, ImageFile
from torchvision.transforms import Resize
from retixfer.train.model.dictionary import definitions
from kornia.augmentation import RandomHorizontalFlip, RandomAffine, ColorJitter

ImageFile.LOAD_TRUNCATED_IMAGES = True

augmentations_pretraining = torch.nn.Sequential(RandomHorizontalFlip(p=0.5),
                                                RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1)),      
                                                ColorJitter(p=0.25, brightness=0.2, contrast=0.2))          

class LoadTensor():
    def __init__(self, target='image_path'):
        self.target = target

    def __call__(self, data):
        file_path = os.path.splitext(data[self.target])[0] + '.pt'
        img = torch.load(file_path)
        data[self.target.replace("_path", "")] = img
        return data

class LoadImage():
    def __init__(self, target="image_path"):
        self.target = target

    def __call__(self, data):
        img = np.array(Image.open(data[self.target]), dtype=float)

        if np.max(img) > 1:
            img /= 255

        if len(img.shape) > 2:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        if img.shape[0] > 3:
            img = img[1:, :, :]

        if "image" in self.target:
            if img.shape[0] < 3:
                img = np.repeat(img, 3, axis=0)

        data[self.target.replace("_path", "")] = img
        return data

class ImageScaling():
    def __init__(self, size=(512, 512), canvas=True, target="image"):
        self.size = size
        self.canvas = canvas                            
        self.target = target

        self.transforms = torch.nn.Sequential(
            Resize(self.size, antialias=True),
        )

    def __call__(self, data):
        img = torch.tensor(data[self.target])

        if not self.canvas or (img.shape[-1] == img.shape[-2]):
            img = self.transforms(img)
        else:
            sizes = img.shape[-2:]                      
            max_size = max(sizes)
            scale = max_size/self.size[0]
            img = Resize((int(img.shape[-2]/scale), int(img.shape[-1]/scale)), antialias=True)(img)
            img = torch.nn.functional.pad(img, (0, self.size[0] - img.shape[-1], 0, self.size[1] - img.shape[-2], 0, 0))

        data[self.target] = img
        return data

class ProduceDescription():
    def __init__(self, caption):
        self.caption = caption

    def __call__(self, data):
        atr_sample = random.sample(data['atributes'], 1)[0] if len(data['atributes']) > 0 else ""
        cat_sample = random.sample(data['categories'], 1)[0] if len(data['categories']) > 0 else ""

        data["sel_category"] = cat_sample
        data["report"] = [self.caption.replace("[ATR]",  atr_sample).replace("[CLS]",  cat_sample).replace("  ", " ")]
        return data


class AugmentDescription():
    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, data):
        if self.augment:
            for category in data["categories"]:
                if category in list(definitions.keys()):
                    prompts = definitions[category]    
                    new_prompt = random.sample(prompts, 1)[0]
                    new_cat = ', '.join([category,new_prompt])
                    data["report"][0] = data["report"][0].replace(category, new_cat)
                    data["augmented_category"] = new_cat
                    
            data["caption"] = data["sel_category"]
            
        return data

class CopyDict():
    def __call__(self, data):
        d = copy.deepcopy(data)
        return d

class SelectRelevantKeys():
    def __init__(self, target_keys=None):
        if target_keys is None:
            target_keys = ['image', 'report', 'sel_category', 'image_path', 'caption']       
        self.target_keys = target_keys

    def __call__(self, data):
        d = {key: data[key] for key in self.target_keys}
        return d