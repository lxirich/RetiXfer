import numpy
import sys

if not hasattr(numpy, "int"):
    numpy.int = int
if not hasattr(numpy, "float"):
    numpy.float = float
if not hasattr(numpy, "bool"):
    numpy.bool = bool

try:
    from numpy.lib import format
except ImportError:
    pass

import numpy.core.multiarray
sys.modules['numpy._core'] = numpy
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

import argparse
import torch
import os

from retixfer.train.data.dataloader import get_loader
from retixfer.train.data.transforms import augmentations_pretraining
from retixfer.train.model.model import RetiXfer
from retixfer.train.model.misc import set_seeds

from local_data.constants import *

import torch
print(torch.__version__, torch.version.cuda)
import torchvision; print(torchvision.__version__)


def process(args):
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    seed = 42
    set_seeds(seed, use_cuda=True)
    
    dataloaders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                             expert_knowledge_path = args.expert_knowledge_path,
                             datasets=args.datasets, batch_size=args.batch_size,
                             num_workers=args.num_workers, banned_categories=args.banned_categories,
                             caption=args.caption, expert_knowledge = args.expert_knowledge)
    
    model = RetiXfer(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=args.load_weights, vision_pretrained=True,
                       weights_path=args.weights_path, text_pretrained = args.text_pretrained, text_weights_path=args.text_weights_path, 
                       n_e=args.n_e, e_dim=args.e_dim, codebook_pretrained=args.codebook_pretrained, 
                       codebook_weights_path=args.codebook_weights_path, exemplar_knowledge_path=args.exemplar_knowledge_path)
    
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.module.fit(dataloaders, transforms=augmentations_pretraining)   


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_path', default=PATH_RESIZED_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN)                         
    parser.add_argument('--datasets', default=[])               
    parser.add_argument('--banned_categories', default=[]) 
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN+"ckpt/", help='output path')
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]")       
    parser.add_argument('--expert_knowledge', default=True, type=lambda x: (str(x).lower() == 'true'))  
    parser.add_argument('--expert_knowledge_path', type=str, default=None)
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v1 -- efficientnet -- ViT_B_512 -- ViT_S_512')    
    parser.add_argument('--n_e', default=16384, type=int)
    parser.add_argument('--e_dim', default=256, type=int)

    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))  
    parser.add_argument('--weights_path', default=None)  
    parser.add_argument('--text_pretrained', default=True, type=lambda x: (str(x).lower() == 'true')) 
    parser.add_argument('--text_weights_path', default=None)
    parser.add_argument('--codebook_pretrained', default=True, type=lambda x: (str(x).lower() == 'true')) 
    parser.add_argument('--codebook_weights_path', default=None)
    parser.add_argument('--exemplar_knowledge_path', default=None)

    parser.add_argument('--store_num', default=1, type=int)                                       
    parser.add_argument('--num_workers', default=4, type=int, help='workers number for DataLoader')
    parser.add_argument("--local_rank", type=int, default=-1)

    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()