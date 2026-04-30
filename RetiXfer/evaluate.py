import argparse
import torch

from retixfer.train.model.model import RetiXfer
from RetiXfer.retixfer.evaluation.dataloader import get_dataloader_splits
from retixfer.utils.metrics import evaluate, average_folds_results, save_results
from retixfer.train.model.misc import set_seeds
from RetiXfer.retixfer.evaluation.adapters import LinearProbe, ClipAdapter, ZeroShot, TipAdapter

from local_data.constants import *

import warnings
warnings.filterwarnings("ignore")

set_seeds(42, use_cuda=torch.cuda.is_available())


def init_adapter(model, args):
    if args.task == "lp":
        print("Evaluation by Linear Probing...", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)
    elif args.task == "clipAdapter":
        print("Evaluation by CLIP Adapter...", end="\n")
        adapter = ClipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge)
    elif args.task == "tipAdapter":
        print("Evaluation by TIP-Adapter Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge, train=False)
    elif args.task == "tipAdapter-f":
        print("Evaluation by TIP-Adapter-f Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge, train=True)
    elif args.task == "zero_shot":
        print("Zero-shot classification...", end="\n")
        adapter = ZeroShot(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge)
    else:
        print("Adapter not implemented... using LP", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)

    return adapter


def generate_experiment_id(args):
    id = args.dataset + args.vision_type + '_task_' + args.task +\
         '_shots_train_' + args.shots_train + '_shots_test_' + args.shots_test + \
         + '_domain knowledge_' + str(args.domain_knowledge) + \
         '_proj_' + str(args.project_features)
    return id


def process(args):
    args.metrics_test, args.metrics_external, args.weights = [], [[] for i in range(len(args.experiment_test))], []    

    experiment_id = generate_experiment_id(args)
    print(experiment_id)

    for iFold in range(args.folds):
        print("\nEvaluation (fold : " + str(iFold + 1) + ")", end="\n")
        args.iFold = iFold

        args.setting = get_experiment_setting(args.dataset)                                                       
        args.loaders = get_dataloader_splits(args.setting["dataframe"], args.data_root_path, args.setting["targets"],
                                             shots_train=args.shots_train, shots_val=args.shots_val,
                                             shots_test=args.shots_test, 
                                             batch_size=args.batch_size, num_workers=args.num_workers, seed=iFold,
                                             task=args.setting["task"], size=args.size,
                                             batch_size_test=args.batch_size_test,
                                             expert_knowledge= args.expert_knowledge, dataset=args.dataset)          
                                                                                                               
        model = RetiXfer(args)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters in model: {total_params}")
        print(f"Trainable parameters in model: {trainable_params}")

        adapter = init_adapter(model, args)                                                                             

        adapter.fit(args.loaders)

        if args.loaders["test"] is not None:
            refs, preds = adapter.predict(args.loaders["test"])
            metrics_fold = evaluate(refs, preds, args.setting["task"])
            args.metrics_test.append(metrics_fold)

        args.weights.append(adapter.model.state_dict())

        if args.experiment_test[0] != "":
            for i_external in range(len(args.experiment_test)):
                print("External testing: " + args.experiment_test[i_external])

                setting_external = get_experiment_setting(args.experiment_test[i_external])
                loaders_external = get_dataloader_splits(setting_external["dataframe"], args.data_root_path,
                                                         args.setting["targets"], shots_train="0%", shots_val="0%",
                                                         shots_test="100%", batch_size=args.batch_size_test,
                                                         batch_size_test=args.batch_size_test,
                                                         num_workers=args.num_workers, seed=iFold,
                                                         task=args.setting["task"], size=args.size,
                                                         resize_canvas=args.resize_canvas)
                
                refs, preds = adapter.predict(loaders_external["test"])
                metrics = evaluate(refs, preds, args.setting["task"])
                args.metrics_external[i_external].append(metrics)

    if args.loaders["test"] is not None:
        print("\nEvaluation (cross-validation)", end="\n")
        args.metrics = average_folds_results(args.metrics_test, args.setting["task"])
    else:
        args.metrics = None

    save_results(args.metrics, args.out_path, id_experiment=generate_experiment_id(args),
                 id_metrics="metrics", save_model=args.save_model, weights=args.weights)

    if args.experiment_test[0] != "":
        for i_external in range(len(args.experiment_test)):
            print("External testing: " + args.experiment_test[i_external])
            metrics = average_folds_results(args.metrics_external[i_external], args.setting["task"])
            save_results(metrics, args.out_path, id_experiment=generate_experiment_id(args),
                         id_metrics=args.experiment_test[i_external], save_model=False)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')   
    parser.add_argument('--shots_train', default="0%", type=lambda x: (str(x)))                   
    parser.add_argument('--shots_val', default="0%", type=lambda x: (str(x)))                           
    parser.add_argument('--shots_test', default="100%", type=lambda x: (str(x)))                           
    parser.add_argument('--folds', default=1, type=int)                                                
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--batch_size_test', default=4, type=int)
    parser.add_argument('--size', default=(512, 512), help="(512, 512) | (2048, 4096) ")
    parser.add_argument('--resize_canvas', default=False, type=lambda x: (str(x).lower() == 'true')) 

    parser.add_argument('--dataset', default='25_REFUGE')            
    parser.add_argument('--experiment_test', default='', type=lambda s: [item for item in s.split(',')])                                           
    parser.add_argument('--task', default='zero_shot')                                                             
    parser.add_argument('--num_workers', default=8, type=int, help='workers number for DataLoader')
    parser.add_argument('--epochs', default=50, type=int)                                      
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--weights_path', default=None)                               
    parser.add_argument('--project_features', default=True, type=lambda x: (str(x).lower() == 'true')) 
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))    
    parser.add_argument('--domain_knowledge', default=False, type=lambda x: (str(x).lower() == 'true')) 
    parser.add_argument('--fta', default=False, type=lambda x: (str(x).lower() == 'true'))             
    parser.add_argument('--tta', default=False, type=lambda x: (str(x).lower() == 'true'))              
    parser.add_argument('--expert_knowledge', default=False, type=lambda x: (str(x).lower() == 'true'))  
    parser.add_argument('--caption', default="A [ATR] fundus photograph of [CLS]") 
    
    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()
