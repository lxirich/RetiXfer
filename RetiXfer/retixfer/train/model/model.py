import ast
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import pandas as pd

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, logging          

from .dictionary import definitions
from . import constants
from .encoders import VisionModel, TextModel


logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(precision=4, sci_mode=False, linewidth=200, threshold=10_000)
np.set_printoptions(precision=4, suppress=True, linewidth=200, threshold=10_000)

global_step = 0

class RetiXfer(torch.nn.Module):
    def __init__(self, vision_type='resnet_v1', bert_type="emilyalsentzer/Bio_ClinicalBERT", vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None, 
                 text_weights_path = None, out_path=None, image_size=512, caption="A fundus photograph of [CLS]", 
                 projection=True, norm_features=True, text_pretrained = True, patch_size = 32, n_e = 100, e_dim = 512, 
                 codebook_weights_path = None, exemplar_knowledge_path=None):
        super().__init__()                           
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.text_weights_path = text_weights_path
        self.codebook_weights_path = codebook_weights_path
        self.exemplar_knowledge_path = exemplar_knowledge_path
        self.out_path = out_path
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained              
        self.text_pretrained = text_pretrained 

        self.image_size = image_size
        self.caption = caption    
        self.projection = projection
        self.norm_features = norm_features
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.n_e = n_e
        self.e_dim = e_dim
        self.logit_scale_init_value = logit_scale_init_value  
        self.patch_size = patch_size

        self.vision_model = VisionModel(vision_type=self.vision_type, patch_size = self.patch_size, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features, text_pretrained = self.text_pretrained, 
                                    text_weights_path = self.text_weights_path)
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        self.load_from_pretrained(self.weights_path)

        self.to(device)

    def load_from_pretrained(self, weights_path=None):
        if weights_path is None:
            import zipfile
            input_dir = constants.PATH_PRETRAINED_WEIGHTS         
            pretrained_id = constants.ID_FLAIR_RESNET_V1           
            pretrained_url_id = constants.URL_ID_FLAIR_RESNET_V1
            weights_path = input_dir + pretrained_id
            if not os.path.exists(input_dir + pretrained_id):
                if not os.path.exists(input_dir):
                    Path(input_dir).mkdir(parents=True, exist_ok=True)

                zipf = zipfile.ZipFile(input_dir + "flair_resnet.zip")
                zipf.extractall(input_dir)
                zipf.close()
                print('\n Download model to:', input_dir + pretrained_id)

        state_dict = torch.load(weights_path, map_location=device)

        self.load_state_dict(state_dict, strict=False)  
        print('load model weight from:', weights_path)

    def load_codebook_from_pretrained(self, weights_path=None):
        state_dict = torch.load(weights_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == 'state_dict':
                for sub_k, sub_v in v.items():
                    if sub_v.shape == torch.Size([self.n_e, self.e_dim]):
                        new_state_dict["embedding.weight"] = sub_v       
        self.quantize.load_state_dict(new_state_dict, strict=True)
        print('load model weight from:', weights_path)

        for param in self.quantize.parameters():
            param.requires_grad = False
            
    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text

    def reduce_tensor(self, tensor: torch.Tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt

    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None, local_rank=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        write = SummaryWriter(log_dir='./results', flush_secs=60)

        if scheduler:
            from RetiXfer.retixfer.train.model.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
        else:
            scheduler = None
            
        epoch = 1
        while epoch <= epochs:
            loss_epoch = self.train_epoch(
                datalaoders["train"], optimizer, scheduler, transforms, epoch, write)

            if local_rank==0:
                print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))
                write.add_scalar("epoch_train_loss", loss_epoch, epoch)
                write.add_scalar("lr", lr, epoch)
                
            if (epoch % store_num == 0) & (local_rank==0):
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.mkdir(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')
            epoch += 1

    def train_epoch(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, write=None):
        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                 
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)                                      

        global global_step  

        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)       

            with torch.autocast(device_type='cuda'):                                               
                if transforms is not None:
                    images = transforms(images)                   
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)
                loss = self.softce_clip_loss(logits_per_text, target).to(device)
                loss = self.reduce_tensor(loss)                         
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_ave += loss.item()

            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) 
                + "- loss_value: " + str(round(loss.item(), 7))
            )
            write.add_scalar("train_loss", loss, global_step)
            
            global_step += 1  

            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)
  
    
    def forward(self, image, text):
        self.eval()
        image = self.preprocess_image(image)
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)
            logits = self.compute_logits(img_embeds, text_embeds).t()
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy(), logits.cpu().numpy()

    def preprocess_image(self, image):
        if image.dtype != np.float32:
            image = np.float32(image)

        if image.max() > 0:
            image /= 255
        if len(image.shape) > 2:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)

        image = torch.tensor(image)
        sizes = image.shape[-2:]
        max_size = max(sizes)
        scale = max_size / self.image_size
        image = torchvision.transforms.Resize((int(image.shape[-2] / scale), int(image.shape[-1] / scale)))(image)
        image = torch.nn.functional.pad(image, (0, self.image_size - image.shape[-1], 0, self.image_size - image.shape[-2], 0, 0))

        image = image.to(torch.float32).to(device)
        return image

    def preprocess_text(self, text):
        prompts = [self.caption.replace("[CLS]", category) for category in text]
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask
    
    def compute_text_embeddings(self, categories, domain_knowledge=False):
        text_embeds_dict = {}                                                       
        for iKey in range(len(categories)):
            if domain_knowledge and categories[iKey] in list(definitions.keys()):  
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]

            with torch.no_grad():
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                print("descriptions",descriptions)
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)
                text_embeds,embed_before = self.text_model(input_ids, attention_mask)            

            text_embeds_dict[categories[iKey]]= text_embeds.mean(0).unsqueeze(0)   

        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds
