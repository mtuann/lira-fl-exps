
import os
import argparse

import torch

from utils import partition_data, seed_experiment, load_poisoned_dataset
import wandb

from models.resnet_tinyimagenet import resnet18
from models.vgg import get_vgg_model
from models.simple import SimpleFLNet
from fl_trainner import test, FrequencyFederatedLearningTrainer
from lira_helper import create_trigger_model

import torch.nn as nn
import copy

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def create_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--based-folder', type=str, default='/home/vinuni/vinuni/user/dung.nt184244/LIRA-Federated-Learning', metavar='N',
                        help='based folder to save and load the results')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    # CHECK HERE
    parser.add_argument('--fraction', type=float or int, default=10,
                        help='how many fraction of poisoned data inserted')
    
    parser.add_argument('--local_train_period', type=int, default=1,
                        help='number of local training epochs')
    parser.add_argument('--num_nets', type=int, default=3383,
                        help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=30,
                        help='number of participating clients per FL round')
    
    parser.add_argument('--fl_round', type=int, default=100,
                        help='total number of FL round to conduct')
    parser.add_argument('--fl_mode', type=str, default="fixed-freq",
                        help='fl mode: fixed-freq mode or fixed-pool mode')
    parser.add_argument('--attacker_pool_size', type=int, default=100,
                        help='size of attackers in the population, used when args.fl_mode == fixed-pool only')    
    parser.add_argument('--defense_method', type=str, default="no-defense",
                        help='defense method used: no-defense|norm-clipping|norm-clipping-adaptive|weak-dp|krum|multi-krum|rfa|')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--attack_method', type=str, default="blackbox",
                        help='describe the attack type: blackbox|pgd|graybox|')
    
     # CHECK HERE
    parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                        help='dataset to use during the training process')
    parser.add_argument('--model', type=str, default='lenet',
                        help='model to use during the training process')  
    
    parser.add_argument('--eps', type=float, default=5e-5,
                        help='specify the l_inf epsilon budget')
    parser.add_argument('--atk_lr', type=float, default=5e-5,
                        help='specify the l_inf epsilon budget')
    parser.add_argument('--norm_bound', type=float, default=3,
                        help='describe if there is defense method: no-defense|norm-clipping|weak-dp|')
    parser.add_argument('--adversarial_local_training_period', type=int, default=5,
                        help='specify how many epochs the adversary should train for')
    parser.add_argument('--poison_type', type=str, default='ardis',
                        help='specify source of data poisoning: |ardis|fashion|(for EMNIST) || |southwest|southwest+wow|southwest-da|greencar-neo|howto|(for CIFAR-10)')
    parser.add_argument('--rand_seed', type=int, default=7,
                        help='random seed utilize in the experiment for reproducibility.')
    parser.add_argument('--model_replacement', type=bool_string, default=False,
                        help='to scale or not to scale')
    parser.add_argument('--project_frequency', type=int, default=10,
                        help='project once every how many epochs')
    parser.add_argument('--adv_lr', type=float, default=0.02,
                        help='learning rate for adv in PGD setting')
    parser.add_argument('--prox_attack', type=bool_string, default=False,
                        help='use prox attack')
    parser.add_argument('--attack_case', type=str, default="edge-case",
                        help='attack case indicates wheather the honest nodes see the attackers poisoned data points: edge-case|normal-case|almost-edge-case')
    parser.add_argument('--stddev', type=float, default=0.158,
                        help='choose std_dev for weak-dp defense')
    parser.add_argument('--attack_portion', type=float, default=1.0,
                        help='Portion to attack the data')
    parser.add_argument('--attack_alpha', type=float, default=0.5,
                        help='Paramter alpha for the optimization of LIRA')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scaling factor for the poisoned model')
    parser.add_argument('--atk_eps', type=float, default=0.01,
                        help='Epsilon for noise constraint in LIRA')
    parser.add_argument('--instance', type=str, default="LIRA-test",
                        help='instance for running wandb instance for easier tracking')
    parser.add_argument('--attack_model', type=str, default="unet",
                        help='model used for conducting the attack (i.e, unet/autoencoder)')
    parser.add_argument('--num_dps_attacker', type=int, default=1000,
                        help='Number of data points for attacker')
    parser.add_argument('--atk_model_train_epoch', type=int, default=1,
                        help='Local training epoch for the attack model')
    parser.add_argument('--target_label', type=int, default=1,
                        help='Target label of backdoor attack settings')
    parser.add_argument('--baseline', type=bool_string, default=False,
                        help='run as baseline')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device(args.device if use_cuda else "cpu")
       
    """
    # hack to make stuff work on GD's machines
    if torch.cuda.device_count() > 2:
        device = 'cuda:4' if use_cuda else 'cpu'
        #device = 'cuda:2' if use_cuda else 'cpu'
        #device = 'cuda' if use_cuda else 'cpu'
    else:
        device = 'cuda' if use_cuda else 'cpu'
    """
    
    return args, kwargs, device
    
def init_lira_config_and_wandb(args):
	# PARSER arguments for LIRA backdoor attack only
    
    # check mode, save_model, clsmodel, target_label
    lira_args = {
        'eps': args.atk_eps,
        'epochs': 50,
        'lr': 0.02,
        'test_eps': None,
        'test_alpha': None,
        'attack_alpha': args.attack_alpha,
        'avoid_cls_reinit': False,
        'mode': 'all2one',
        'lr_atk': args.atk_lr,
        'save_model': False,
        'clsmodel': args.model,
        'train_epoch': args.atk_model_train_epoch,
        'attack_model': args.attack_model,
        'attack_portion': args.attack_portion,
        'path': 'saved_path/',
        'target_label': args.target_label,
        'dataset': args.dataset,
        'best_threshold': 0.1
    }
    
    instance_name = "LIRA-first-trial" # check not use
    
    group_name = "Idea-Trials"
    
    wandb_ins_name = f"baseline_{args.baseline}_{args.dataset}_alpha_{args.attack_alpha}_eps_{args.atk_eps}_numiter_{args.adversarial_local_training_period}_{args.attack_portion}_epc_{args.atk_model_train_epoch}_{args.model}"
    
    # wandb_ins = wandb.init(project="LIRA in FL", 
    #                        entity="vinuni-ai-secure-lab",
    #                         name=wandb_ins_name,
    #                         group=group_name)

    # lira_args = namedtuple('Struct', lira_args.keys())(*lira_args.values())
    
    wandb_ins = None
    
    return lira_args, wandb_ins

def get_args_for_fl_training(args, net_avg, net_dataidx_map, poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader, scratch_model, atkmodel, tgtmodel):
    arguments = {}
    
    if args.fl_mode == "fixed-freq":
        arguments = {
            #"poisoned_emnist_dataset":poisoned_emnist_dataset,
            "vanilla_model":vanilla_model,
            "net_avg":net_avg,
            "net_dataidx_map":net_dataidx_map,
            "num_nets":args.num_nets,
            "dataset":args.dataset,
            "model":args.model,
            "part_nets_per_round":args.part_nets_per_round,
            "fl_round":args.fl_round,
            "local_training_period":args.local_train_period, #5 #1
            "adversarial_local_training_period":args.adversarial_local_training_period,
            "args_lr":args.lr,
            "args_gamma":args.gamma,
            "baseline":args.baseline,
            "atk_eps":args.atk_eps,
            "scale":args.scale,
            # "attacking_fl_rounds":[i for i in range(1, args.fl_round + 1) if (i-1)%10 == 0], #"attacking_fl_rounds":[i for i in range(1, fl_round + 1)], #"attacking_fl_rounds":[1],
            #"attacking_fl_rounds":[i for i in range(1, args.fl_round + 1) if (i-1)%100 == 0], #"attacking_fl_rounds":[i for i in range(1, fl_round + 1)], #"attacking_fl_rounds":[1],
            "attacking_fl_rounds":[i for i in range(1, args.fl_round + 1)], # one attacker participating each training round
            "num_dps_poisoned_dataset":num_dps_poisoned_dataset,
            "poisoned_emnist_train_loader":poisoned_train_loader,
            "clean_train_loader":clean_train_loader,
            "vanilla_emnist_test_loader":vanilla_test_loader,
            "targetted_task_test_loader":targetted_task_test_loader,
            "batch_size":args.batch_size,
            "test_batch_size":args.test_batch_size,
            "log_interval":args.log_interval,
            "defense_technique":args.defense_method,
            "attack_method":args.attack_method,
            "eps":args.eps,
            "atk_lr":args.atk_lr,
            "norm_bound":args.norm_bound,
            "poison_type":args.poison_type,
            "device":device,
            "model_replacement":args.model_replacement,
            "project_frequency":args.project_frequency,
            "adv_lr":args.adv_lr,
            "prox_attack":args.prox_attack,
            "attack_case":args.attack_case,
            "stddev":args.stddev,
            "atkmodel": atkmodel,
            "tgtmodel": tgtmodel,
            # "create_net": create_net,
            "scratch_model": scratch_model,
            "based_folder":args.based_folder,
        }

        
    elif args.fl_mode == "fixed-pool":
        arguments = {
            #"poisoned_emnist_dataset":poisoned_emnist_dataset,
            "vanilla_model":vanilla_model,
            "net_avg":net_avg,
            "net_dataidx_map":net_dataidx_map,
            "num_nets":args.num_nets,
            "dataset":args.dataset,
            "model":args.model,
            "part_nets_per_round":args.part_nets_per_round,
            "attacker_pool_size":args.attacker_pool_size,
            "fl_round":args.fl_round,
            "local_training_period":args.local_train_period,
            "adversarial_local_training_period":args.adversarial_local_training_period,
            "args_lr":args.lr,
            "args_gamma":args.gamma,
            "num_dps_poisoned_dataset":num_dps_poisoned_dataset,
            "poisoned_emnist_train_loader":poisoned_train_loader,
            "clean_train_loader":clean_train_loader,
            "vanilla_emnist_test_loader":vanilla_test_loader,
            "targetted_task_test_loader":targetted_task_test_loader,
            "batch_size":args.batch_size,
            "test_batch_size":args.test_batch_size,
            "log_interval":args.log_interval,
            "defense_technique":args.defense_method,
            "attack_method":args.attack_method,
            "eps":args.eps,
            "norm_bound":args.norm_bound,
            "poison_type":args.poison_type,
            "device":device,
            "model_replacement":args.model_replacement,
            "project_frequency":args.project_frequency,
            "adv_lr":args.adv_lr,
            "prox_attack":args.prox_attack,
            "attack_case":args.attack_case,
            "stddev":args.stddev,
            "based_folder":args.based_folder,
    }
    return arguments
    

def get_global_model(args, READ_CKPT=True):
    
    if READ_CKPT:
        if args.model == "lenet":
            net_avg = SimpleFLNet(num_classes=10).to(device)
            with open(f"{args.based_folder}/checkpoint/emnist_lenet_10epoch.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
                
        elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
            net_avg = get_vgg_model(args.model).to(device)
            # net_avg = VGG(args.model.upper()).to(device)
            # load model here
            #with open("./checkpoint/trained_checkpoint_vanilla.pt", "rb") as ckpt_file:
            with open(f"{args.based_folder}/checkpoint/Cifar10_{args.model.upper()}_10epoch.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
                
        elif args.model in ("resnet18tiny"):

            net_avg = resnet18(num_classes=200).to(device)
            ckpt_state_dict = torch.load(f"{args.based_folder}/checkpoint/tiny-resnet.epoch_20", map_location=device)["state_dict"]
             
            # with open("./checkpoint/tiny-resnet.epoch_20".format(args.model.upper()), "rb") as ckpt_file:
                # ckpt_state_dict = torch.load(ckpt_file['state_dict'], map_location=device)
                
        net_avg.load_state_dict(ckpt_state_dict)
        
        logger.info("Loading checkpoint file successfully ...")
        
    else:
        if args.model == "lenet":
            net_avg = SimpleFLNet(num_classes=10).to(device)
        elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
            net_avg = get_vgg_model(args.model).to(device)
        elif args.model in ("resnet18tiny"):
            net_avg = resnet18(num_classes=200).to(device)
            
    scratch_model = copy.deepcopy(net_avg)
    logger.info("Test the model performance on the entire task before FL process ... ")
    return net_avg, scratch_model


if __name__ == "__main__":
    
     
    args, kwargs, device = create_parser()
    
    args.model = 'resnet18tiny'
    
    lira_args, wandb_ins = init_lira_config_and_wandb(args)
    
    logger.info("Running LIRA backdoor attack in FL with args: {}".format(args))
    logger.info("Running LIRA backdoor attack in FL with lira_args: {}".format(lira_args))
    logger.info(f"wandb_ins: {wandb_ins} kwargs: {kwargs} device: {device}")
    
    logger.info(device)
    
    logger.info('==> Building model..')
    
    atkmodel, tgtmodel = create_trigger_model(args.dataset, device)
    
    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    # add random seed for the experiment for reproducibility
    seed_experiment(seed=args.rand_seed, logger=logger)

    
    # the hyper-params are inspired by the paper "Can you really backdoor FL?" (https://arxiv.org/pdf/1911.07963.pdf)
    # partition_strategy = "homo"
    partition_strategy = "hetero-dir"
    # print("Process partition_data function")
    dir_alpha = 0.01 if args.dataset == "tiny-imagenet" else 0.5
    print(args.num_nets)
    net_dataidx_map = partition_data(
            args.dataset, '/home/vinuni/vinuni/user/dung.nt184244/LIRA-Federated-Learning/data/', partition_strategy,
            args.num_nets, dir_alpha, args) # 0.5 for cifar10, mnist; 0.01 for tinyimagenet
    print("Finish partition_data function")
    
    
    
    # rounds of fl to conduct
    ## some hyper-params here:
    local_training_period = args.local_train_period #5 #1
    adversarial_local_training_period = 5

    poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader = load_poisoned_dataset(args=args, kwargs=kwargs, logger=logger)
    
    # vanilla_test_loader, clean_train_loader --> for 1 client --> for tiny img
    # clean_train_loader --> poisoned data loader, belong to attackers. 
    # net_avg: global model
    
    # TuanNM: check local_train_period
    
    net_avg, scratch_model = get_global_model(args, READ_CKPT=True)

    # test(logger, net_avg, device, vanilla_test_loader, test_batch_size=args.test_batch_size, criterion=criterion, mode="raw-task", dataset=args.dataset)
    
    # test(logger, net_avg, device, targetted_task_test_loader, test_batch_size=args.test_batch_size, criterion=criterion, mode="targetted-task", dataset=args.dataset, poison_type=args.poison_type)

    # let's remain a copy of the global model for measuring the norm distance:
    vanilla_model = copy.deepcopy(net_avg)
    arguments_fl_training = get_args_for_fl_training(args, net_avg, net_dataidx_map, poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader, scratch_model, atkmodel, tgtmodel)
    if args.fl_mode == 'fixed-freq':
        print(f"Start running fixed-freq FL mode with {args.num_nets} clients")
        frequency_fl_trainer = FrequencyFederatedLearningTrainer(logger=logger, arguments=arguments_fl_training, lira_args=lira_args)
        frequency_fl_trainer.run(wandb_ins = wandb_ins)
        
    elif args.fl_mode == "fixed-pool":
        pass
        # fixed_pool_fl_trainer = FixedPoolFederatedLearningTrainer(arguments=arguments)
        # fixed_pool_fl_trainer.run()