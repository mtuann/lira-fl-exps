

import random
import torch
import numpy as np
import os
import torch
from torchvision import transforms
from datasets import MNIST_truncated, EMNIST_truncated, CIFAR10_truncated, ImageFolderTruncated


def seed_experiment(seed=0, logger=None):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything")



def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_emnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    emnist_train_ds = EMNIST_truncated(datadir, train=True, download=True, transform=transform)
    emnist_test_ds = EMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = emnist_train_ds.data, emnist_train_ds.target
    X_test, y_test = emnist_test_ds.data, emnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def partition_data(dataset, datadir, partition, n_nets, alpha, args):
    print("Start partition_data")
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_data(datadir)
        n_train = X_train.shape[0]
        
    elif dataset == 'tiny-imagenet':
        print(f"Check dataset: {dataset}")
   
        _train_dir = f'{args.based_folder}/data/tiny-imagenet-200/train'
        _val_dir = f'{args.based_folder}/data/tiny-imagenet-200/val'
            
        tiny_mean = [0.485, 0.456, 0.406]
        tiny_std = [0.229, 0.224, 0.225]
        
        _data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ]),
        }
        
        trainset = ImageFolderTruncated(_train_dir, transform=_data_transforms['train'])
        y_train = trainset.get_train_labels
        
        n_train = y_train.shape[0]
        
    elif dataset.lower() == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        print("Start partition hetero-dir")
        min_size = 0
        # K = 200
        from collections import Counter
        counter_class = Counter(y_train)
        # print(f"Counter class for dataset: {dataset}: {counter_class}")
        K = len(counter_class)
        N = y_train.shape[0]
        net_dataidx_map = {}
        
        # print(N)
        # import IPython
        # IPython.embed()
        
        while (min_size < 10) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        
        # for j in range(n_nets):
        #     # print("Number of data points in client {}: {}".format(j, ))
        #     y_class_j = y_train[net_dataidx_map[j]]
        #     print("Number of data points in each class in client {} total {} {}".format(j, len(net_dataidx_map[j]), Counter(y_class_j)))
        
        # print("Number of data points for all clients: {}".format(sum([len(net_dataidx_map[j]) for j in range(n_nets)])))
        # print("---"*30)
        
        # print("Debug IPython")
        # import IPython
        # IPython.embed()
        
        if dataset == 'cifar10':
            if args.poison_type == 'howto' or args.poison_type == 'greencar-neo':
                green_car_indices = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
                #sanity_check_counter = 0
                for k, v in net_dataidx_map.items():
                    remaining_indices = [i for i in v if i not in green_car_indices]
                    #sanity_check_counter += len(remaining_indices)
                    net_dataidx_map[k] = remaining_indices

            #logger.info("Remaining total number of data points : {}".format(sanity_check_counter))
            # sanity check:
            #aggregated_val = []
            #for val in net_dataidx_map.values():
            #    aggregated_val+= val
            #black_box_indices = [i for i in range(50000) if i not in aggregated_val]
            #logger.info("$$$$$$$$$$$$$$ recovered black box indices: {}".format(black_box_indices))
            #exit()
    # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    print("Done partition hetero-dir")
    
    return net_dataidx_map


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'emnist', 'cifar10', 'tiny-imagenet'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        if dataset == 'emnist':
            dl_obj = EMNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])
            
        elif dataset == 'tiny-imagenet':

            _data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ]),
            }
        
            dl_obj = ImageFolderTruncated

        
        
            transform_train = _data_transforms['train']

            transform_test = _data_transforms['val']
            
        if dataset == 'tiny-imagenet':
            
            _train_dir = f'{datadir}/tiny-imagenet-200/train'
            _val_dir = f'{datadir}/tiny-imagenet-200/val'
            
            train_ds = dl_obj(root=_train_dir, dataidxs=dataidxs, transform=transform_train)
            test_ds = dl_obj(root=_val_dir, transform=transform_test)
            
        else:
            train_ds = dl_obj(datadir, train=True, dataidxs=dataidxs, transform=transform_train, download=True)
            test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
            
        train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl



def get_dataloader_normal_case(dataset, datadir, train_bs, test_bs, 
                                dataidxs=None, 
                                user_id=0, 
                                num_total_users=200,
                                poison_type="southwest",
                                ardis_dataset=None,
                                attack_case='normal-case'):
    if dataset in ('mnist', 'emnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        if dataset == 'emnist':
            dl_obj = EMNIST_NormalCase_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        elif dataset == 'cifar10':
            dl_obj = CIFAR10NormalCase_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

        # this only supports cifar10 right now, please be super careful when calling it using other datasets
        # def __init__(self, root, 
        #                 dataidxs=None, 
        #                 train=True, 
        #                 transform=None, 
        #                 target_transform=None, 
        #                 download=False,
        #                 user_id=0,
        #                 num_total_users=200,
        #                 poison_type="southwest"):        
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True,
                                    user_id=user_id, num_total_users=num_total_users, poison_type=poison_type,
                                    ardis_dataset_train=ardis_dataset, attack_case=attack_case)
        
        test_ds = None #dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def load_poisoned_dataset(args, kwargs, logger):
    
    num_sampled_data_points = args.num_dps_attacker
    
    if args.dataset in ("mnist", "emnist"):
        if args.fraction < 1:
            fraction=args.fraction  #0.1 #10
        else:
            fraction=int(args.fraction)

        # prepare MNIST dataset
        mnist_train_dataset = datasets.MNIST(root=f'{args.based_folder}/data', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
        mnist_test_dataset = datasets.MNIST(root=f'{args.based_folder}/data', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
        total_training_samples = len(mnist_train_dataset)
        samped_data_indices = np.random.choice(total_training_samples, num_sampled_data_points, replace=False)
                    # samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
        mnist_train_dataset.data = mnist_train_dataset.data[samped_data_indices, :, :]
        mnist_train_dataset.targets = np.array(mnist_train_dataset.targets)[samped_data_indices]
        
        poisoned_train_loader = None
        vanilla_test_loader = torch.utils.data.DataLoader(mnist_test_dataset,
             batch_size=args.test_batch_size, shuffle=False, **kwargs)
        targetted_task_test_loader = None
        clean_train_loader = torch.utils.data.DataLoader(mnist_train_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)
        
        num_dps_poisoned_dataset = mnist_train_dataset.data.shape[0]
    
    elif args.dataset == "tiny-imagenet":
        # TUANNM: TODO: load tiny-imagenet dataset
        vanilla_test_loader = None
        _train_dir = f'{args.based_folder}/data/tiny-imagenet-200/train'
        _val_dir = f'{args.based_folder}/data/tiny-imagenet-200/val'
        _data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
            ]),
        }
        tiny_mean=[0.485, 0.456, 0.406],
        tiny_std=[0.229, 0.224, 0.225],
        
        trainset = ImageFolderTruncated(_train_dir, transform = _data_transforms['train'])
        
        valset =   ImageFolderTruncated(_val_dir, transform = _data_transforms['val'])
        
        vanilla_test_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        samped_data_indices = np.random.choice(len(trainset), num_sampled_data_points, replace=False)
        # poisoned_trainset = trainset[samped_data_indices, :, :, :]
        truncated_train_set = ImageFolderTruncated(_train_dir, dataidxs=samped_data_indices, transform=_data_transforms['train'] )
        
        clean_train_loader = torch.utils.data.DataLoader(truncated_train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        
        poisoned_train_loader = clean_train_loader
        
        targetted_task_test_loader, num_dps_poisoned_dataset = None, num_sampled_data_points
        
        
    elif args.dataset == "cifar10":
        
        # TUANNM: Check if the dataset is poisoned or not
        poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader = get_data_poisoned_for_cifar10(args, logger)
        
        
    return poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader

def get_data_poisoned_for_cifar10(args, logger):
    based_folder = args.based_folder
    
    if args.poison_type == "southwest":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = torchvision.datasets.CIFAR10(root=f'{based_folder}/data', train=True, download=True, transform=transform_train)

        poisoned_trainset = copy.deepcopy(trainset)

        if args.attack_case == "edge-case":
            with open(f'{based_folder}/saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)

            with open(f'{based_folder}/saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)
        elif args.attack_case == "normal-case" or args.attack_case == "almost-edge-case":
            with open(f'{based_folder}/saved_datasets/southwest_images_adv_p_percent_edge_case.pkl', 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)

            with open(f'{based_folder}/saved_datasets/southwest_images_p_percent_edge_case_test.pkl', 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)
        else:
            raise NotImplementedError("Not Matched Attack Case ...")             

        #
        logger.info("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
        #sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
        sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
        
        logger.info("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
        #sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
        sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck



        # downsample the poisoned dataset #################
        if args.attack_case == "edge-case":
            num_sampled_poisoned_data_points = 100 # N
            samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                            num_sampled_poisoned_data_points,
                                                            replace=False)
            saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
            sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
            logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
        elif args.attack_case == "normal-case" or args.attack_case == "almost-edge-case":
            num_sampled_poisoned_data_points = 100 # N
            samped_poisoned_data_indices = np.random.choice(784,
                                                            num_sampled_poisoned_data_points,
                                                            replace=False)
        ######################################################


        # downsample the raw cifar10 dataset #################
        # num_sampled_data_points = 1000 # M
        samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
        poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
        poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
        logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
        # keep a copy of clean data
        clean_trainset = copy.deepcopy(poisoned_trainset)
        ########################################################


        poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
        poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

        logger.info("{}".format(poisoned_trainset.data.shape))
        logger.info("{}".format(poisoned_trainset.targets.shape))
        logger.info("{}".format(sum(poisoned_trainset.targets)))


        #poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root=f'{based_folder}/data', train=False, download=True, transform=transform_test)

        poisoned_testset = copy.deepcopy(testset)
        poisoned_testset.data = saved_southwest_dataset_test
        poisoned_testset.targets = sampled_targets_array_test

        # vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        # targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
        targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)

        num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]
    elif args.poison_type == "southwest-da":
        
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        transform_poison = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            AddGaussianNoise(0., 0.05),
            ])            
        # data prep for test set
        transform_test = transforms.Compose([transforms.ToTensor(),normalize])

        #transform_test = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = torchvision.datasets.CIFAR10(root=f'{args.based_folder}/data', train=True, download=True, transform=transform_train)

        #poisoned_trainset = copy.deepcopy(trainset)
        #  class CIFAR10_Poisoned(data.Dataset):
        #def __init__(self, root, clean_indices, poisoned_indices, dataidxs=None, train=True, transform_clean=None,
        #    transform_poison=None, target_transform=None, download=False):

        with open(f'{based_folder}/saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)

        with open(f'{based_folder}/saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_southwest_dataset_test = pickle.load(test_f)

        #
        logger.info("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
        sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
        
        logger.info("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
        sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck



        # downsample the poisoned dataset ###########################
        num_sampled_poisoned_data_points = 100 # N
        samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                        num_sampled_poisoned_data_points,
                                                        replace=False)
        saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
        sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
        logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
        ###############################################################


        # downsample the raw cifar10 dataset #################
        num_sampled_data_points = 1000 # M
        samped_data_indices = np.random.choice(trainset.data.shape[0], num_sampled_data_points, replace=False)
        tempt_poisoned_trainset = trainset.data[samped_data_indices, :, :, :]
        tempt_poisoned_targets = np.array(trainset.targets)[samped_data_indices]
        logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
        ########################################################

        poisoned_trainset = CIFAR10_Poisoned(root=f'{based_folder}/data', 
                            clean_indices=np.arange(tempt_poisoned_trainset.shape[0]), 
                            poisoned_indices=np.arange(tempt_poisoned_trainset.shape[0], tempt_poisoned_trainset.shape[0]+saved_southwest_dataset_train.shape[0]), 
                            train=True, download=True, transform_clean=transform_train,
                            transform_poison=transform_poison)
        #poisoned_trainset = CIFAR10_truncated(root=f'{args.based_folder}/data', dataidxs=None, train=True, transform=transform_train, download=True)
        clean_trainset = copy.deepcopy(poisoned_trainset)

        poisoned_trainset.data = np.append(tempt_poisoned_trainset, saved_southwest_dataset_train, axis=0)
        poisoned_trainset.target = np.append(tempt_poisoned_targets, sampled_targets_array_train, axis=0)

        logger.info("{}".format(poisoned_trainset.data.shape))
        logger.info("{}".format(poisoned_trainset.target.shape))


        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
        clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root=f'{based_folder}/data', train=False, download=True, transform=transform_test)

        poisoned_testset = copy.deepcopy(testset)
        poisoned_testset.data = saved_southwest_dataset_test
        poisoned_testset.targets = sampled_targets_array_test

        vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
        targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)

        num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]    
    elif args.poison_type == "howto":
        """
        implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = torchvision.datasets.CIFAR10(root=f'{args.based_folder}/data', train=True, download=True, transform=transform_train)

        poisoned_trainset = copy.deepcopy(trainset)

        ##########################################################################################################################
        sampled_indices_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
                                19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389]
        sampled_indices_test = [32941, 36005, 40138]
        cifar10_whole_range = np.arange(trainset.data.shape[0])
        remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_train+sampled_indices_test]
        logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(len(sampled_indices_train+sampled_indices_test)))
        saved_greencar_dataset_train = trainset.data[sampled_indices_train, :, :, :]
        #########################################################################################################################

        # downsample the raw cifar10 dataset ####################################################################################
        num_sampled_data_points = 500-len(sampled_indices_train)
        samped_data_indices = np.random.choice(remaining_indices, num_sampled_data_points, replace=False)
        poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
        poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
        logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
        clean_trainset = copy.deepcopy(poisoned_trainset)
        ##########################################################################################################################

        # we load the test since in the original paper they augment the 
        with open(f'{args.based_folder}/saved_datasets/green_car_transformed_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        #
        logger.info("Backdoor (Green car) train-data shape we collected: {}".format(saved_greencar_dataset_train.shape))
        sampled_targets_array_train = 2 * np.ones((saved_greencar_dataset_train.shape[0],), dtype =int) # green car -> label as bird
        
        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = 2 * np.ones((saved_greencar_dataset_test.shape[0],), dtype =int) # green car -> label as bird/


        poisoned_trainset.data = np.append(poisoned_trainset.data, saved_greencar_dataset_train, axis=0)
        poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

        logger.info("Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape))
        logger.info("Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape))


        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
        clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root=f'{args.based_folder}/data', train=False, download=True, transform=transform_test)

        poisoned_testset = copy.deepcopy(testset)
        poisoned_testset.data = saved_greencar_dataset_test
        poisoned_testset.targets = sampled_targets_array_test

        vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
        targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)
        num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]
    elif args.poison_type == "greencar-neo":
        """
        implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = torchvision.datasets.CIFAR10(root=f'{args.based_folder}/data', train=True, download=True, transform=transform_train)

        poisoned_trainset = copy.deepcopy(trainset)

        with open(f'{args.based_folder}/saved_datasets/new_green_cars_train.pkl', 'rb') as train_f:
            saved_new_green_cars_train = pickle.load(train_f)

        with open(f'{args.based_folder}/saved_datasets/new_green_cars_test.pkl', 'rb') as test_f:
            saved_new_green_cars_test = pickle.load(test_f)

        # we use the green cars in original cifar-10 and new collected green cars
        ##########################################################################################################################
        num_sampled_poisoned_data_points = 100 # N
        sampled_indices_green_car = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
                                19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
        cifar10_whole_range = np.arange(trainset.data.shape[0])
        remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_green_car]
        #ori_cifar_green_cars = trainset.data[sampled_indices_green_car, :, :, :]

        samped_poisoned_data_indices = np.random.choice(saved_new_green_cars_train.shape[0],
                                                        #num_sampled_poisoned_data_points-len(sampled_indices_green_car),
                                                        num_sampled_poisoned_data_points,
                                                        replace=False)
        saved_new_green_cars_train = saved_new_green_cars_train[samped_poisoned_data_indices, :, :, :]

        #saved_greencar_dataset_train = np.append(ori_cifar_green_cars, saved_new_green_cars_train, axis=0)
        saved_greencar_dataset_train = saved_new_green_cars_train
        logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(saved_greencar_dataset_train.shape[0]))
        #########################################################################################################################

        # downsample the raw cifar10 dataset ####################################################################################
        num_sampled_data_points = 400
        samped_data_indices = np.random.choice(remaining_indices, num_sampled_data_points, replace=False)
        poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
        poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
        logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
        clean_trainset = copy.deepcopy(poisoned_trainset)
        ##########################################################################################################################

        #
        logger.info("Backdoor (Green car) train-data shape we collected: {}".format(saved_greencar_dataset_train.shape))
        sampled_targets_array_train = 2 * np.ones((saved_greencar_dataset_train.shape[0],), dtype =int) # green car -> label as bird
        
        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_new_green_cars_test.shape))
        sampled_targets_array_test = 2 * np.ones((saved_new_green_cars_test.shape[0],), dtype =int) # green car -> label as bird/


        poisoned_trainset.data = np.append(poisoned_trainset.data, saved_greencar_dataset_train, axis=0)
        poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

        logger.info("Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape))
        logger.info("Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape))


        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
        clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root=f'{args.based_folder}/data', train=False, download=True, transform=transform_test)

        poisoned_testset = copy.deepcopy(testset)
        poisoned_testset.data = saved_new_green_cars_test
        poisoned_testset.targets = sampled_targets_array_test

        vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
        targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)
        num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

    return poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader

