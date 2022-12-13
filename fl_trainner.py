import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from termcolor import colored
import numpy as np
import copy

from lira_helper import get_target_transform, get_clip_image, create_paths
from utils import get_dataloader, get_dataloader_normal_case

def calc_norm_diff(gs_model, vanilla_model, epoch, fl_round, mode="bad", logger=None):
    norm_diff = 0
    for p_index, p in enumerate(gs_model.parameters()):
        norm_diff += torch.norm(list(gs_model.parameters())[p_index] - list(vanilla_model.parameters())[p_index]) ** 2
    norm_diff = torch.sqrt(norm_diff).item()
    if mode == "bad":
        #pdb.set_trace()
        logger.info("===> ND `|w_bad-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "normal":
        logger.info("===> ND `|w_normal-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))
    elif mode == "avg":
        logger.info("===> ND `|w_avg-w_g|` in local epoch: {} | FL round: {} |, is {}".format(epoch, fl_round, norm_diff))

    return norm_diff

def fed_avg_aggregator(init_model, net_list, net_freq, device, model="lenet"):
   
        

    # import IPython
    # IPython.embed()
    
    weight_accumulator = {}
    
    for name, params in init_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params).float()
    
    for i in range(0, len(net_list)):
        diff = dict()
        for name, data in net_list[i].state_dict().items():
            # diff[name] = (data - model_server_before_aggregate.state_dict()[name]).cpu().detach().numpy()
            diff[name] = (data - init_model.state_dict()[name])
            try:
                weight_accumulator[name].add_(net_freq[i]  *  diff[name])
                # weight_accumulator[name].add_(0.1  *  diff[name])
                
            except Exception as e:
                print(e)
                import IPython
                IPython.embed()
                exit(0)
    
    for idl, (name, data) in enumerate(init_model.state_dict().items()):
        update_per_layer = weight_accumulator[name] #  * self.conf["lambda"]
        
        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))
            # data.add_(update_per_layer.float())
            
        else:
            data.add_(update_per_layer)
            # print(idl, name, torch.sum(data - net_list[0].state_dict()[name]))
            
    # import IPython
    # IPython.embed()

    return init_model

def test(logger, model, device, test_loader, test_batch_size, criterion, mode="raw-task", dataset="cifar10", poison_type="fashion"):
    number_class = 10
    if dataset == 'tiny-imagenet':
        number_class = 200
        
    class_correct = list(0. for i in range(number_class))
    class_total = list(0. for i in range(number_class))

        
    
    if dataset in ("mnist", "emnist"):
        target_class = 7
        if mode == "raw-task":
            classes = [str(i) for i in range(10)]
        elif mode == "targetted-task":
            if poison_type == 'ardis':
                classes = [str(i) for i in range(10)]
            else: 
                classes = ["T-shirt/top", 
                            "Trouser",
                            "Pullover",
                            "Dress",
                            "Coat",
                            "Sandal",
                            "Shirt",
                            "Sneaker",
                            "Bag",
                            "Ankle boot"]
                
    elif dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # target_class = 2 for greencar, 9 for southwest
        if poison_type in ("howto", "greencar-neo"):
            target_class = 2
        else:
            target_class = 9
            
    elif dataset == "tiny-imagenet":
        classes = [str(i) for i in range(200)]
        target_class = 0
        # TuanNM: check here
      
    model.eval()
    test_loss = 0
    correct = 0
    backdoor_correct = 0
    backdoor_tot = 0
    final_acc = 0
    task_acc = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check backdoor accuracy
            if poison_type == 'ardis':
                backdoor_index = torch.where(target == target_class)
                target_backdoor = torch.ones_like(target[backdoor_index])
                predicted_backdoor = predicted[backdoor_index]
                backdoor_correct += (predicted_backdoor == target_backdoor).sum().item()
                backdoor_tot = backdoor_index[0].shape[0]
                # logger.info("Target: {}".format(target_backdoor))
                # logger.info("Predicted: {}".format(predicted_backdoor))

            #for image_index in range(test_batch_size):
            for image_index in range(len(target)):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                    
                class_total[label] += 1
    test_loss /= len(test_loader.dataset)
    

        
    if mode == "raw-task":
        for i in range(number_class):
            logger.info('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

            if i == target_class:
                task_acc = 100 * class_correct[i] / class_total[i]

        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        final_acc = 100. * correct / len(test_loader.dataset)

    elif mode == "targetted-task":

        if dataset in ("mnist", "emnist"):
            for i in range(10):
                logger.info('Accuracy of %5s : %.2f %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
            if poison_type == 'ardis':
                # ensure 7 is being classified as 1
                logger.info('Backdoor Accuracy of %.2f : %.2f %%' % (
                     target_class, 100 * backdoor_correct / backdoor_tot))
                final_acc = 100 * backdoor_correct / backdoor_tot
            else:
                # trouser acc
                final_acc = 100 * class_correct[1] / class_total[1]
        
        elif dataset == "cifar10":
            logger.info('#### Targetted Accuracy of %5s : %.2f %%' % (classes[target_class], 100 * class_correct[target_class] / class_total[target_class]))
            final_acc = 100 * class_correct[target_class] / class_total[target_class]
    return final_acc, task_acc


def test_updated(args, device, atkmodel, model, target_transform, 
         train_loader, test_loader, epoch, trainepoch, clip_image, 
         testoptimizer=None, log_prefix='Internal', epochs_per_test=3, 
         dataset="cifar10", criterion=None, subpath_saved=""):

    if args['test_alpha'] is None:
        args['test_alpha'] = args['attack_alpha']
    if args['test_eps'] is None:
        args['test_eps'] = args['eps']
        
    num_class = 10
    if dataset == 'tiny-imagenet':
        num_class = 200
        
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))
    
    
    model.eval()
    test_loss = 0
    correct = 0
    backdoor_correct = 0
    backdoor_tot = 0
    final_acc = 0
    task_acc = None
    

    correct = 0    
    correct_transform = 0
    test_loss = 0
    test_transform_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            bs = data.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * bs  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            noise = atkmodel(data) * args['test_eps']
            atkdata = clip_image(data + noise)
            atkoutput = model(atkdata)
            test_transform_loss += criterion(atkoutput, target_transform(target)).item() * bs  # sum up batch loss
            atkpred = atkoutput.max(1, keepdim=True)[
                1]  # get the index of the max log-probability
            correct_transform += atkpred.eq(
                target_transform(target).view_as(atkpred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_transform_loss /= len(test_loader.dataset)

    correct /= len(test_loader.dataset)
    correct_transform /= len(test_loader.dataset)
    print(f"\nTest result without retraining: ")
    print(
        '{}-Test set [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.2f} poison {:.2f}'.format(
            log_prefix, 0, 
            test_loss, test_transform_loss,
            correct, correct_transform
        ))
    #     writer.add_image(f"{log_prefix}-Test Images", grid, global_step=(epoch-1))
    if subpath_saved and not os.path.exists(f"track_trigger/{subpath_saved}"):
        os.makedirs(f"track_trigger/{subpath_saved}")
    if epoch % epochs_per_test == 0 and log_prefix == 'External' and subpath_saved:
        batch_img = torch.cat(
        [data[:4].clone().cpu(), noise[:4].clone().cpu(), atkdata[:4].clone().cpu()], 0)
        batch_img = F.upsample(batch_img, scale_factor=(4, 4))
        grid = torchvision.utils.make_grid(batch_img, normalize=True)
        torchvision.utils.save_image(grid, f"track_trigger/{subpath_saved}/ckp_trigger_image_epoch_{epoch}.png")

    return correct, correct_transform
    
def train(model, device, train_loader, optimizer, epoch, log_interval, criterion, pgd_attack=False, eps=5e-4, model_original=None,
        proj="l_2", project_frequency=1, adv_optimizer=None, prox_attack=False, wg_hat=None, logger=None):
    """
        train function for both honest nodes and adversary.
        NOTE: this trains only for one epoch
    """
    model.train()
    # get learning rate
    for param_group in optimizer.param_groups:
        eta = param_group['lr']

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if pgd_attack:
            adv_optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        if prox_attack:
            wg_hat_vec = parameters_to_vector(list(wg_hat.parameters()))
            model_vec = parameters_to_vector(list(model.parameters()))
            prox_term = torch.norm(wg_hat_vec - model_vec)**2
            loss = loss + prox_term
        
        loss.backward()
        if not pgd_attack:
            optimizer.step()
        else:
            if proj == "l_inf":
                w = list(model.parameters())
                n_layers = len(w)
                # adversarial learning rate
                eta = 0.001
                for i in range(len(w)):
                    # uncomment below line to restrict proj to some layers
                    if True:#i == 6 or i == 8 or i == 10 or i == 0 or i == 18:
                        w[i].data = w[i].data - eta * w[i].grad.data
                        # projection step
                        m1 = torch.lt(torch.sub(w[i], model_original[i]), -eps)
                        m2 = torch.gt(torch.sub(w[i], model_original[i]), eps)
                        w1 = (model_original[i] - eps) * m1
                        w2 = (model_original[i] + eps) * m2
                        w3 = (w[i]) * (~(m1+m2))
                        wf = w1+w2+w3
                        w[i].data = wf.data
            else:
                # do l2_projection
                adv_optimizer.step()
                w = list(model.parameters())
                w_vec = parameters_to_vector(w)
                model_original_vec = parameters_to_vector(model_original)
                # make sure you project on last iteration otherwise, high LR pushes you really far
                if (batch_idx%project_frequency == 0 or batch_idx == len(train_loader)-1) and (torch.norm(w_vec - model_original_vec) > eps):
                    # project back into norm ball
                    w_proj_vec = eps*(w_vec - model_original_vec)/torch.norm(
                            w_vec-model_original_vec) + model_original_vec
                    # plug w_proj back into model
                    vector_to_parameters(w_proj_vec, w)
                # for i in range(n_layers):
                #    # uncomment below line to restrict proj to some layers
                #    if True:#i == 16 or i == 17:
                #        w[i].data = w[i].data - eta * w[i].grad.data
                #        if torch.norm(w[i] - model_original[i]) > eps/n_layers:
                #            # project back to norm ball
                #            w_proj= (eps/n_layers)*(w[i]-model_original[i])/torch.norm(
                #                w[i]-model_original[i]) + model_original[i]
                #            w[i].data = w_proj

        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


class FederatedLearningTrainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()
    
    
class FrequencyFederatedLearningTrainer(FederatedLearningTrainer):
    def __init__(self, logger, arguments=None, lira_args=None, *args, **kwargs):
        #self.poisoned_emnist_dataset = arguments['poisoned_emnist_dataset']
        self.logger = logger
        self.vanilla_model = arguments['vanilla_model']
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_period = arguments['local_training_period']
        self.adversarial_local_training_period = arguments['adversarial_local_training_period']
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.attacking_fl_rounds = arguments['attacking_fl_rounds']
        self.poisoned_emnist_train_loader = arguments['poisoned_emnist_train_loader']
        self.clean_train_loader = arguments['clean_train_loader']
        self.vanilla_emnist_test_loader = arguments['vanilla_emnist_test_loader']
        self.targetted_task_test_loader = arguments['targetted_task_test_loader']
        self.batch_size = arguments['batch_size']
        self.test_batch_size = arguments['test_batch_size']
        self.log_interval = arguments['log_interval']
        self.device = arguments['device']
        # self.num_dps_poisoned_dataset = int(arguments['num_dps_poisoned_dataset'] * (1.0+lira_args['attack_portion']))
        self.num_dps_poisoned_dataset = arguments['num_dps_poisoned_dataset']
        
        self.defense_technique = arguments["defense_technique"]
        self.norm_bound = arguments["norm_bound"]
        self.attack_method = arguments["attack_method"]
        self.dataset = arguments["dataset"]
        self.model = arguments["model"]
        self.criterion = nn.CrossEntropyLoss()
        self.eps = arguments['eps']
        self.poison_type = arguments['poison_type']
        self.model_replacement = arguments['model_replacement']
        self.project_frequency = arguments['project_frequency']
        self.adv_lr = arguments['adv_lr']
        self.atk_lr = arguments['atk_lr']
        self.prox_attack = arguments['prox_attack']
        self.attack_case = arguments['attack_case']
        self.stddev = arguments['stddev']
        self.atkmodel = arguments['atkmodel']
        self.tgtmodel = arguments['tgtmodel']
        self.lira_args = lira_args
        self.baseline = arguments['baseline']
        # self.create_net = arguments['create_net']
        self.scratch_model = arguments['scratch_model']
        self.atk_eps = arguments['atk_eps']
        self.scale_factor = arguments['scale']
        self.based_folder = arguments['based_folder']

        if self.attack_method == "pgd":
            self.pgd_attack = True
        else:
            self.pgd_attack = False

        # if arguments["defense_technique"] == "no-defense":
        #     self._defender = None
        # elif arguments["defense_technique"] == "norm-clipping" or arguments["defense_technique"] == "norm-clipping-adaptive":
        #     self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        # elif arguments["defense_technique"] == "weak-dp":
        #     # doesn't really add noise. just clips
        #     self._defender = WeightDiffClippingDefense(norm_bound=arguments['norm_bound'])
        # elif arguments["defense_technique"] == "krum":
        #     self._defender = Krum(mode='krum', num_workers=self.part_nets_per_round, num_adv=1)
        # elif arguments["defense_technique"] == "multi-krum":
        #     self._defender = Krum(mode='multi-krum', num_workers=self.part_nets_per_round, num_adv=1)
        # elif arguments["defense_technique"] == "rfa":
        #     self._defender = RFA()
        # else:
        #     NotImplementedError("Unsupported defense method !")

    def run(self, wandb_ins):
        logger = self.logger
        
        main_task_acc = []
        raw_task_acc = []
        backdoor_task_acc = []
        fl_iter_list = []
        adv_norm_diff_list = []
        wg_norm_list = []
        
        # variables for LIRA algorithms only
        trainlosses = []
        best_acc_clean = 0
        best_acc_poison = 0
        avoid_cls_reinit = True
        clip_image = get_clip_image(self.dataset)
        attack_train_epoch = self.lira_args['train_epoch']

        target_transform = get_target_transform(self.lira_args)
        tgt_optimizer = None
        loss_list = []
        
        basepath, checkpoint_path, bestmodel_path = create_paths(self.lira_args)
        print('========== PATHS ==========')
        print(f'Basepath: {basepath}')
        print(f'Checkpoint Model: {checkpoint_path}')
        print(f'Best Model: {bestmodel_path}')
        
        LOAD_ATK_MODEL = False
        if os.path.exists(checkpoint_path) and LOAD_ATK_MODEL:
            #Load previously saved models
            checkpoint = torch.load(checkpoint_path)
            print(colored('Load existing attack model from path {}'.format(checkpoint_path), 'red'))
            self.atkmodel.load_state_dict(checkpoint['atkmodel'], strict=True)
            # clsmodel.load_state_dict(checkpoint['clsmodel'], strict=True)
            trainlosses = checkpoint['trainlosses']
            best_acc_clean = checkpoint['best_acc_clean']
            best_acc_poison = checkpoint['best_acc_poison']
            start_epoch = checkpoint['epoch']
            tgt_optimizer.load_state_dict(checkpoint['tgtoptimizer'])
        else:
            #Create new model
            print(colored('Create new model from {}'.format(checkpoint_path), 'blue'))
            best_acc_clean = 0
            best_acc_poison = 0
            trainlosses = []
            start_epoch = 1
            
        subpath_trigger_saved = f"{self.dataset}baseline_{self.baseline}_atkepc_{attack_train_epoch}_eps_{self.atk_eps}"
        
        if self.baseline:
            subpath_trigger_saved = ""
            
        acc_clean, backdoor_acc = test_updated(self.lira_args, self.device, self.atkmodel, self.net_avg, target_transform, 
                                self.clean_train_loader, self.vanilla_emnist_test_loader, 0, self.lira_args['train_epoch'], clip_image, 
                                testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, subpath_saved=subpath_trigger_saved)
        
        print(f"\n----------TEST FOR GLOBAL MODEL BEFORE FEDERATED TRAINING----------------")
        print(f"Main task acc: {round(acc_clean*100, 2)}%")
        print(f"Backdoor task acc: {round(backdoor_acc*100, 2)}%")
        print(f"--------------------------------------------------------------------------\n")
        
        local_acc_clean, local_acc_poison = 0.0, 0.0
        
        # let's conduct multi-round training
        for flr in range(1, self.fl_round+1):
            # logger.info("##### attack fl rounds: {}".format(self.attacking_fl_rounds))
            g_user_indices = []

            if self.defense_technique == "norm-clipping-adaptive":
                # experimental
                norm_diff_collector = []

            if flr in self.attacking_fl_rounds and not self.baseline:
                # randomly select participating clients
                
                
                selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round-1, replace=False)
                num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
                total_num_dps_per_round = sum(num_data_points) + self.num_dps_poisoned_dataset

                logger.info("FL round: {}, total num data points: {}, num dps poisoned: {}".format(flr, num_data_points, self.num_dps_poisoned_dataset))

                net_freq = [self.num_dps_poisoned_dataset/ total_num_dps_per_round] + [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round-1)]
                logger.info("Net freq: {}, FL round: {} with adversary".format(net_freq, flr)) 
                #pdb.set_trace()

                # we need to reconstruct the net list at the beginning
                net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
                logger.info("################## Starting fl round: {}".format(flr))
                
                model_original = list(self.net_avg.parameters())
                # super hacky but I'm doing this for the prox-attack
                wg_clone = copy.deepcopy(self.net_avg)
                wg_hat = None
                v0 = torch.nn.utils.parameters_to_vector(model_original)
                wg_norm_list.append(torch.norm(v0).item())
               
                # start the FL process
                for net_idx, net in enumerate(net_list):
                    #net  = net_list[net_idx]                
                    if net_idx == 0:
                        global_user_idx = -1 # we assign "-1" as the indices of the attacker in global user indices
                        pass
                    else:
                        global_user_idx = selected_node_indices[net_idx-1]
                        dataidxs = self.net_dataidx_map[global_user_idx]
                        if self.attack_case == "edge-case":
                            # print(f"Load data from folder edge-case: {self.based_folder}/data")
                            train_dl_local, _ = get_dataloader(self.dataset, f'{self.based_folder}/data', self.batch_size, 
                                                            self.test_batch_size, dataidxs) # also get the data loader
                            
                        elif self.attack_case in ("normal-case", "almost-edge-case"):
                            train_dl_local, _ = get_dataloader_normal_case(self.dataset, f'{self.based_folder}/data', self.batch_size, 
                                                            self.test_batch_size, dataidxs, user_id=global_user_idx,
                                                            num_total_users=self.num_nets,
                                                            poison_type=self.poison_type,
                                                            ardis_dataset=self.ardis_dataset,
                                                            attack_case=self.attack_case) # also get the data loader
                        else:
                            NotImplementedError("Unsupported attack case ...")

                    g_user_indices.append(global_user_idx)
                    if net_idx == 0:
                        logger.info("@@@@@@@@ Working on client: {}, which is Attacker".format(net_idx))
                    else:
                        logger.info("@@@@@@@@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                    adv_optimizer = optim.SGD(net.parameters(), lr=self.adv_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # looks like adversary needs same lr to hide with others
                    prox_optimizer = optim.SGD(wg_clone.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
                    tgt_optimizer = optim.Adam(self.tgtmodel.parameters(), lr=self.atk_lr)
                    atk_optimizer = optim.Adam(self.atkmodel.parameters(), lr=self.atk_lr)
                    
                    # duplicate_model = copy.deepcopy(net)
                    # duplicate_optimizer = optim.SGD(duplicate_model.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4)
                    for param_group in optimizer.param_groups:
                        logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))
                    
                    if net_idx == 0:
                        # Local training on data of an attacker
                        if self.prox_attack:
                            # estimate w_hat
                            for inner_epoch in range(1, self.local_training_period+1):
                                estimate_wg(wg_clone, self.device, self.clean_train_loader, prox_optimizer, inner_epoch, log_interval=self.log_interval, criterion=self.criterion)
                            wg_hat = wg_clone
                            
                        for e in range(1, self.adversarial_local_training_period+1):
                           # we always assume net index 0 is adversary
                            if self.defense_technique in ('krum', 'multi-krum'):
                                train(net, self.device, self.poisoned_emnist_train_loader, optimizer, e, log_interval=self.log_interval, criterion=self.criterion,
                                        pgd_attack=self.pgd_attack, eps=self.eps*self.args_gamma**(flr-1), model_original=model_original, project_frequency=self.project_frequency, adv_optimizer=adv_optimizer,
                                        prox_attack=self.prox_attack, wg_hat=wg_hat, logger=logger)
                            else:
                                # Focus on this case, we temporarily leave out any available defenses
                                logger.info(f"\n\nSTART LOCAL TRAINING FOR ATTACKER LOCAL ITERATION [{e}/{self.adversarial_local_training_period}]")
                                train_local_model(net, self.atkmodel, self.device,  self.clean_train_loader, adv_optimizer, self.criterion, proj="l_2", project_frequency=1, 
                                                atk_eps = self.atk_eps, prox_attack=False, wg_hat=None, clip_image=clip_image, target_transform = target_transform, 
                                                flr=flr, attack_portion=self.lira_args['attack_portion'], train_epoch=2)
                                train_atk_model(net, self.atkmodel, self.device,  self.clean_train_loader, atk_optimizer, self.criterion,  atk_eps = self.atk_eps, 
                                                attack_alpha = self.lira_args['attack_alpha'], flr=flr, attack_portion=self.lira_args['attack_portion'], 
                                                train_epoch=attack_train_epoch, clip_image=clip_image, target_transform=target_transform)
                                
                               
                                # self.atkmodel.load_state_dict(self.tgtmodel.state_dict())
                                self.tgtmodel.load_state_dict(self.atkmodel.state_dict())
                                # net.load_state_dict(scratch_model.state_dict())
                            
                                
                                ma, ba = 0.0, 0.0
                                
                                                            
                            # if not self.lira_args['avoid_cls_reinit']:
                            #     clsmodel = self.scratch_model.to(self.device)
                            #     scratchmodel = self.scratch_model.to(self.device)
                            # else:
                            #     print(f"Transfering for testing on local device!!!")
                            #     scratchmodel = self.scratch_model.to(self.device)
                            #     scratchmodel.load_state_dict(net.state_dict()) #transfer from cls to scratch for testing
        
                            # if epoch % args.epochs_per_external_eval == 0 or epoch == args.epochs: 
                            #     acc_clean, acc_poison = test(args, atkmodel, scratchmodel, target_transform, 
                            #         train_loader, test_loader, epoch, args.cls_test_epochs, writer, clip_image, 
                            #         log_prefix='External')
                            # else:
                            local_acc_clean, local_acc_poison = test_updated(self.lira_args, self.device, self.atkmodel, net, target_transform, 
                                self.clean_train_loader, self.vanilla_emnist_test_loader, e, self.lira_args['train_epoch'], clip_image,
                                log_prefix='Internal', dataset=self.dataset, criterion=self.criterion, subpath_saved=subpath_trigger_saved)

                         
                        # if model_replacement scale models
                        if self.model_replacement:
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))
                            # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                            # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

                            for idx, param in enumerate(net.parameters()):
                                param.data = (param.data - model_original[idx])*(total_num_dps_per_round/self.num_dps_poisoned_dataset) + model_original[idx]
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))
                            
                        if self.scale_factor != 1.0:
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker before scaling : Norm = {}".format(torch.norm(v)))
                            # adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad")
                            # logger.info("||w_bad - w_avg|| before scaling = {}".format(adv_norm_diff))

                            for idx, param in enumerate(net.parameters()):
                                param.data = param.data*self.scale_factor
                            v = torch.nn.utils.parameters_to_vector(net.parameters())
                            logger.info("Attacker after scaling : Norm = {}".format(torch.norm(v)))

                        # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
                        # we can print the norm diff out for debugging
                        adv_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="bad", logger=logger)
                        adv_norm_diff_list.append(adv_norm_diff)

                        if self.defense_technique == "norm-clipping-adaptive":
                            # experimental
                            norm_diff_collector.append(adv_norm_diff)
                    else:
                        for e in range(1, self.local_training_period+1):
                           train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion, logger=logger)              
                           # at here we can check the distance between w_normal and w_g i.e. `\|w_bad - w_g\|_2`
                        # we can print the norm diff out for debugging
                        honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal", logger=logger)
                        
                        if self.defense_technique == "norm-clipping-adaptive":
                            # experimental
                            norm_diff_collector.append(honest_norm_diff)            

            else:
                # in this current version, we sample `part_nets_per_round-1` per FL round since we assume attacker will always participates
                
                selected_node_indices = np.random.choice(self.num_nets, size=self.part_nets_per_round, replace=False)
                num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
                total_num_dps_per_round = sum(num_data_points)

                net_freq = [num_data_points[i]/total_num_dps_per_round for i in range(self.part_nets_per_round)]
                logger.info("Net freq: {}, FL round: {} without adversary".format(net_freq, flr))

                # we need to reconstruct the net list at the beginning
                net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
                logger.info("################## Starting fl round: {}".format(flr))

                # start the FL process
                for net_idx, net in enumerate(net_list):
                    global_user_idx = selected_node_indices[net_idx]
                    dataidxs = self.net_dataidx_map[global_user_idx] # --> keep the same data for an attacker 

                    if self.attack_case == "edge-case":
                        
                        train_dl_local, _ = get_dataloader(self.dataset, f'{self.based_folder}/data', self.batch_size, 
                                                       self.test_batch_size, dataidxs) # also get the data loader
                        
                    elif self.attack_case in ("normal-case", "almost-edge-case"):
                        train_dl_local, _ = get_dataloader_normal_case(self.dataset, f'{self.based_folder}/data', self.batch_size, 
                                                            self.test_batch_size, dataidxs, user_id=global_user_idx,
                                                            num_total_users=self.num_nets,
                                                            poison_type=self.poison_type,
                                                            ardis_dataset=self.ardis_dataset,
                                                            attack_case=self.attack_case) # also get the data loader
                    else:
                        NotImplementedError("Unsupported attack case ...")

                    g_user_indices.append(global_user_idx)
                    
                    logger.info("@@@@@@@@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=self.args_lr*self.args_gamma**(flr-1), momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion
                    
                    for param_group in optimizer.param_groups:
                        logger.info("Effective lr in FL round: {} is {}".format(flr, param_group['lr']))

                    for e in range(1, self.local_training_period+1):
                        train(net, self.device, train_dl_local, optimizer, e, log_interval=self.log_interval, criterion=self.criterion, logger=logger)

                    # honest_norm_diff = calc_norm_diff(gs_model=net, vanilla_model=self.net_avg, epoch=e, fl_round=flr, mode="normal", logger=logger)

                    # if self.defense_technique == "norm-clipping-adaptive":
                    #     # experimental
                    #     norm_diff_collector.append(honest_norm_diff)   

                adv_norm_diff_list.append(0)
                model_original = list(self.net_avg.parameters())
                v0 = torch.nn.utils.parameters_to_vector(model_original)
                wg_norm_list.append(torch.norm(v0).item())


            # ### conduct defense here:
            # if self.defense_technique == "no-defense":
            #     pass
            # elif self.defense_technique == "norm-clipping":
            #     for net_idx, net in enumerate(net_list):
            #         self._defender.exec(client_model=net, global_model=self.net_avg)
            # elif self.defense_technique == "norm-clipping-adaptive":
            #     # we will need to adapt the norm diff first before the norm diff clipping
            #     logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}".format(norm_diff_collector, 
            #         np.mean(norm_diff_collector)))
            #     self._defender.norm_bound = np.mean(norm_diff_collector)
            #     for net_idx, net in enumerate(net_list):
            #         self._defender.exec(client_model=net, global_model=self.net_avg)
            # elif self.defense_technique == "weak-dp":
            #     # this guy is just going to clip norm. No noise added here
            #     for net_idx, net in enumerate(net_list):
            #         self._defender.exec(client_model=net,
            #                             global_model=self.net_avg,)
            # elif self.defense_technique == "krum":
            #     net_list, net_freq = self._defender.exec(client_models=net_list, 
            #                                             num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
            #                                             g_user_indices=g_user_indices,
            #                                             device=self.device)
            # elif self.defense_technique == "multi-krum":
            #     net_list, net_freq = self._defender.exec(client_models=net_list, 
            #                                             num_dps=[self.num_dps_poisoned_dataset]+num_data_points,
            #                                             g_user_indices=g_user_indices,
            #                                             device=self.device)
            # elif self.defense_technique == "rfa":
            #     net_list, net_freq = self._defender.exec(client_models=net_list,
            #                                             net_freq=net_freq,
            #                                             maxiter=500,
            #                                             eps=1e-5,
            #                                             ftol=1e-7,
            #                                             device=self.device)
            # else:
            #     NotImplementedError("Unsupported defense method !")

            # after local training periods
            
            print(f"\nStart performing fed_avg_aggregator...")
            
            self.net_avg = fed_avg_aggregator(self.net_avg, net_list, net_freq, device=self.device, model=self.model)

            # acc_clean, backdoor_acc = test_updated(args=self.lira_args, device=self.device, atkmodel=self.tgtmodel, model=self.net_avg, target_transform=target_transform, 
            #                         train_loader=self.clean_train_loader, test_loader=self.vanilla_emnist_test_loader,epoch=flr, trainepoch=self.lira_args['train_epoch'], clip_image=clip_image, 
            #                         testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, subpath_saved=subpath_trigger_saved)
            


            # v = torch.nn.utils.parameters_to_vector(self.net_avg.parameters())
            # logger.info("############ Averaged Model : Norm {}".format(torch.norm(v)))

            # calc_norm_diff(gs_model=self.net_avg, vanilla_model=self.vanilla_model, epoch=0, fl_round=flr, mode="avg", logger=logger)
            
            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))

            
            acc_clean, backdoor_acc = test_updated(self.lira_args, self.device, self.tgtmodel, self.net_avg, target_transform, 
                                    self.clean_train_loader, self.vanilla_emnist_test_loader, flr, self.lira_args['train_epoch'], clip_image, 
                                    testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, subpath_saved=subpath_trigger_saved)
            
            print(f"\n----------Testing for global model after aggregation:----------------")
            print(f"Main task acc: {round(acc_clean*100, 2)}%")
            print(f"Backdoor task acc: {round(backdoor_acc*100, 2)}%")
            
            
            
            
            acc_clean, backdoor_acc = test_updated(self.lira_args, self.device, self.tgtmodel, net_list[0], target_transform, 
                                    self.clean_train_loader, self.vanilla_emnist_test_loader, flr, self.lira_args['train_epoch'], clip_image, 
                                    testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, subpath_saved=subpath_trigger_saved)
            
            print(f"\n----------Testing for MODEL FROM CLIENT 00 after aggregation:----------------")
            print(f"Main task acc: {round(acc_clean*100, 2)}%")
            print(f"Backdoor task acc: {round(backdoor_acc*100, 2)}%")
            
            print(f"-----------------------------------------------------\n")
            
            
            acc_clean, backdoor_acc = test_updated(self.lira_args, self.device, self.tgtmodel, net_list[1], target_transform, 
                                    self.clean_train_loader, self.vanilla_emnist_test_loader, flr, self.lira_args['train_epoch'], clip_image, 
                                    testoptimizer=None, log_prefix='External', epochs_per_test=3, dataset=self.dataset, criterion=self.criterion, subpath_saved=subpath_trigger_saved)
            
            print(f"\n----------Testing for MODEL FROM CLIENT 01 after aggregation:----------------")
            print(f"Main task acc: {round(acc_clean*100, 2)}%")
            print(f"Backdoor task acc: {round(backdoor_acc*100, 2)}%")
            
            print(f"-----------------------------------------------------\n")
            
            # if acc_clean > best_acc_clean or (acc_clean+self.lira_args['best_threshold'] > best_acc_clean and best_acc_poison < backdoor_acc):
            #     best_acc_poison = backdoor_acc
            #     best_acc_clean = acc_clean
            #     # torch.save({'atkmodel': self.atkmodel.state_dict(), 'clsmodel': self.net_avg.state_dict()}, bestmodel_path)
            
            # wandb_logging_items = {
            #     'fl_iter': flr,
            #     'main_task_acc': acc_clean*100.0,
            #     'backdoor_task_acc': backdoor_acc*100.0, 
            #     'local_best_acc_clean': best_acc_clean,
            #     'local_best_acc_poison': best_acc_poison,
            #     'local_MA': local_acc_clean*100.0,
            #     'local_BA': local_acc_poison*100.0
            # }
            
            # if wandb_ins:
            #     wandb_ins.log({"General Information": wandb_logging_items})
            
            # raw_acc = 0
            # overall_acc = acc_clean
            # fl_iter_list.append(flr)
            # main_task_acc.append(overall_acc)
            # raw_task_acc.append(raw_acc)
            # backdoor_task_acc.append(backdoor_acc)

        # df = pd.DataFrame({'fl_iter': fl_iter_list, 
        #                     'main_task_acc': main_task_acc, 
        #                     'backdoor_acc': backdoor_task_acc, 
        #                     'raw_task_acc':raw_task_acc, 
        #                     'adv_norm_diff': adv_norm_diff_list, 
        #                     'wg_norm': wg_norm_list
        #                     })
       
        # if self.poison_type == 'ardis':
        #     # add a row showing initial accuracies
        #     df1 = pd.DataFrame({'fl_iter': [0], 'main_task_acc': [88], 'backdoor_acc': [11], 'raw_task_acc': [0], 'adv_norm_diff': [0], 'wg_norm': [0]})
        #     df = pd.concat([df1, df])

        # results_filename = get_results_filename(self.poison_type, self.attack_method, self.model_replacement, self.project_frequency,
        #         self.defense_technique, self.norm_bound, self.prox_attack, False, self.model)

        # df.to_csv(results_filename, index=False)
        
        # logger.info("Wrote accuracy results to: {}".format(results_filename))

        # save model net_avg
        # torch.save(self.net_avg.state_dict(), "./checkpoint/emnist_lenet_10epoch.pt")