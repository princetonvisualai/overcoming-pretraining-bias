from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import cv2
import re
import copy
import pickle
import json
import torch
from torch.autograd import Variable
import torch.multiprocessing
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from scipy import ndimage
from datasets import *
from utils import *
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, r2_score
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from ray.tune.search.basic_variant import BasicVariantGenerator

OI_BIN_OUTPUTS = ['openimages_v500kw14_9', 'openimages_v500kw14_10', 'openimages_v500kw14_11', 'openimages_v500kw14_12', 'openimages_v500kw14_13']

def load_data(args):
    workers = 4
    test_transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    train_transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        ])
    
    dontreduce = False # set if I am manually picking out the smaller train num

    if 'coco' in args.dataset:
        if args.dataset == 'coco_6cat':
            train_dataset = CoCoDataset(train_transform, cat6=True)
            val_dataset = CoCoDataset(test_transform, cat6=True)
            test_dataset = CoCoDataset(test_transform, version='val', cat6=True)
            out_features = 6
        elif 'salient' in args.dataset or 'skew' in args.dataset:
            train_dataset = CoCoDataset(train_transform, variation=args.dataset.split('_')[1])
            val_dataset = CoCoDataset(test_transform, variation=args.dataset.split('_')[1])
            if args.testreg:
                if 'salient' in args.dataset and 'skew' in args.dataset:
                    test_dataset = CoCoDataset(test_transform, version='val', variation=args.dataset.split('_')[1].split('+')[0])
                else:
                    test_dataset = CoCoDataset(test_transform, version='val')
            else:
                test_dataset = CoCoDataset(test_transform, version='val', variation=args.dataset.split('_')[1])
            out_features = 80
        elif 'coco_obj' == args.dataset[:8]:
            dontreduce = True
            out_features = 1
            assert args.train_num in [1000, 5000]
            if '+' in args.dataset:
                obj, setnum = args.dataset[8:].split('+')
                setnum = int(setnum[3:])
            else:
                obj = args.dataset[8:]
                setnum = None
            train_dataset = CoCoDataset(train_transform, obj=obj, setnum=setnum, train_num=args.train_num)
            val_dataset = CoCoDataset(test_transform, obj=obj, setnum=setnum, train_num=args.train_num)
            test_dataset = CoCoDataset(test_transform, version='val', obj=obj)

            train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
            all_train_ids = train_dataset.image_ids
            train_dataset.image_ids = all_train_ids[train_indices]
        elif 'dsset' in args.dataset:
            setnum = args.dataset[10:]
            num = int(args.train_num/.8)+5
            train_dataset = CoCoDataset(train_transform, setnum=setnum, ds=True, train_num=num)
            val_dataset = CoCoDataset(test_transform, setnum=setnum, ds=True, train_num=num)
            test_dataset = CoCoDataset(test_transform, version='val')
            if not args.hp_search:
                args.dollarstreet = True
            out_features = 80
        else:
            pred_gen = False
            out_features = 80
            if args.dataset == 'coco_gender':
                pred_gen = True
                out_features = 1
            train_dataset = CoCoDataset(train_transform, pred_gen=pred_gen)
            val_dataset = CoCoDataset(test_transform, pred_gen=pred_gen)
            test_dataset = CoCoDataset(test_transform, version='val', pred_gen=pred_gen)

        if args.train_num != -1 and args.dataset == 'coco':
            train_dataset.image_ids = pickle.load(open('dataset_files/imageids_{}.pkl'.format(num_to_name(args.train_num)), 'rb'))

        if dontreduce:
            val_ids = np.array(list(set(val_dataset.image_ids).difference(set(train_dataset.image_ids))))
            val_dataset.image_ids = val_ids
        else:
            train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
            all_train_ids = train_dataset.image_ids
            train_dataset.image_ids = all_train_ids[train_indices]
            val_dataset.image_ids = all_train_ids[val_indices]
        criterion = nn.BCELoss(reduction='none')

        if args.dollarstreet:
            dollarstreet_dataset = DollarStreetDataset(test_transform, version='test')
            if 'dsset' in args.dataset: ## remove whatever is in the train and val set
                ds_imageids = np.concatenate([[imageid[2:] for imageid in train_dataset.image_ids], [imageid[2:] for imageid in val_dataset.image_ids]])
                dollarstreet_dataset.image_ids = np.array(list(set(dollarstreet_dataset.image_ids).difference(set(ds_imageids))))

            dollarstreet_loader = torch.utils.data.DataLoader(dollarstreet_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=workers)

    elif 'openimages' in args.dataset:
        train_dataset = OpenImagesDataset(train_transform, version='train_v{}'.format(args.dataset[12:]))
        val_dataset = OpenImagesDataset(test_transform, version='train_v{}'.format(args.dataset[12:]))
        if args.testreg:
            assert '500kw14_' in args.dataset
            test_dataset = OpenImagesDataset(test_transform, version='test_with_gen_v{}'.format('500kw14_0'))
        else:
            test_dataset = OpenImagesDataset(test_transform, version='test_with_gen_v{}'.format(args.dataset[12:]))

        if not args.test:
            train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
            all_train_ids = train_dataset.image_ids
            train_dataset.image_ids = all_train_ids[train_indices]
            val_dataset.image_ids = all_train_ids[val_indices]

        out_features = 600
        criterion = nn.BCELoss(reduction='none')

        if args.dataset in OI_BIN_OUTPUTS:
            out_features = 1
            test_dataset.version = 'test_with_gen_v{}'.format(args.dataset[12:])
    elif args.dataset == 'oi_gender':
        train_dataset = OpenImagesGenderDataset(train_transform, version='train')
        val_dataset = OpenImagesGenderDataset(test_transform, version='train')
        test_dataset = OpenImagesGenderDataset(test_transform, version='test')

        train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
        all_train_ids = train_dataset.image_ids
        train_dataset.image_ids = all_train_ids[train_indices]
        val_dataset.image_ids = all_train_ids[val_indices]

        out_features = 3
        criterion = nn.BCELoss(reduction='none')
    elif 'fairface' in args.dataset:
        if '-' in args.dataset:
            name, only = args.dataset.split('-')
            train_dataset = FairFaceDataset(train_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), only=only)
            val_dataset = FairFaceDataset(test_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), only=only)
            test_dataset = FairFaceDataset(test_transform, version='val_{}'.format('_'.join(name.split('_')[1:])), only=only)
        elif 'set' in args.dataset[-5:]:
            if 'set10' == args.dataset[-5:]:
                name, setnum = args.dataset[:-6], args.dataset[-5:]
            elif args.dataset[-4:-1] == 'set':
                name, setnum = args.dataset[:-5], args.dataset[-4:]
            else:
                assert NotImplementedError
            train_dataset = FairFaceDataset(train_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), setnum=setnum)
            val_dataset = FairFaceDataset(test_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), setnum=setnum)
            test_dataset = FairFaceDataset(test_transform, version='val_{}'.format('_'.join(name.split('_')[1:])))
        else:
            train_dataset = FairFaceDataset(train_transform, version='train_{}'.format('_'.join(args.dataset.split('_')[1:])))
            val_dataset = FairFaceDataset(test_transform, version='train_{}'.format('_'.join(args.dataset.split('_')[1:])))
            test_dataset = FairFaceDataset(test_transform, version='val_{}'.format('_'.join(args.dataset.split('_')[1:])))

        train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
        all_train_ids = train_dataset.image_ids
        train_dataset.image_ids = all_train_ids[train_indices]
        val_dataset.image_ids = all_train_ids[val_indices]

        out_features = 1
        criterion = nn.BCELoss(reduction='none')
    elif 'celeba' in args.dataset:
        if '&' in args.dataset:
            train_dataset = CelebASaliencyDataset(train_transform, version='train_{}'.format('_'.join(args.dataset.split('_')[1:])))
            val_dataset = CelebASaliencyDataset(test_transform, version='train_{}'.format('_'.join(args.dataset.split('_')[1:])))
            test_dataset = CelebASaliencyDataset(test_transform, version='val_{}'.format('_'.join(args.dataset.split('_')[1:])))
        elif 'set' in args.dataset[-10:]:
            name, setnum = args.dataset[:args.dataset.index('set')-1], args.dataset[args.dataset.index('set'):]
            if '+' in name:
                dontreduce = True
                train_dataset = CelebADataset(train_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), setnum=setnum, trainnum=args.train_num)
                val_dataset = CelebADataset(test_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), setnum=setnum)
            else:
                train_dataset = CelebADataset(train_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), setnum=setnum)
                val_dataset = CelebADataset(test_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), setnum=setnum)
            test_dataset = CelebADataset(test_transform, version='val_{}'.format('_'.join(name.split('_')[1:])))
        elif '-' in args.dataset:
            name, only = args.dataset.split('-')
            train_dataset = CelebADataset(train_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), only=only)
            val_dataset = CelebADataset(test_transform, version='train_{}'.format('_'.join(name.split('_')[1:])), only=only)
            test_dataset = CelebADataset(test_transform, version='val_{}'.format('_'.join(name.split('_')[1:])))
        else:
            train_dataset = CelebADataset(train_transform, version='train_{}'.format('_'.join(args.dataset.split('_')[1:])))
            val_dataset = CelebADataset(test_transform, version='train_{}'.format('_'.join(args.dataset.split('_')[1:])))
            test_dataset = CelebADataset(test_transform, version='val_{}'.format('_'.join(args.dataset.split('_')[1:])))

        if dontreduce:
            val_ids = np.array(list(set(val_dataset.image_ids).difference(set(train_dataset.image_ids))))
            val_dataset.image_ids = val_ids
        else:
            train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
            all_train_ids = train_dataset.image_ids
            train_dataset.image_ids = all_train_ids[train_indices]
            val_dataset.image_ids = all_train_ids[val_indices]

        out_features = 1
        if 'Rotate' in args.dataset:
            out_features = 4
        criterion = nn.BCELoss(reduction='none')
        if args.fairface:
            fairface_dataset = FairFaceDataset(test_transform, version='train_Female')
            fairface_loader = torch.utils.data.DataLoader(fairface_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=workers)

    elif 'dollarstreet' in args.dataset:
        setnum = None
        if '_' in args.dataset:
            setnum = args.dataset.split('_')[1]
        train_dataset = DollarStreetDataset(train_transform, version='train', setnum=setnum)
        val_dataset = DollarStreetDataset(test_transform, version='train', setnum=setnum)
        test_dataset = DollarStreetDataset(test_transform, version = 'test')

        train_indices, test_indices = train_test_split(np.arange(len(test_dataset.image_ids)), test_size=0.4, random_state=3)
        train_indices = np.array(list(set(train_dataset.image_ids)&set(test_dataset.image_ids[train_indices])))
        train_indices, val_indices = train_test_split(train_indices, test_size=0.2)
        all_train_ids = test_dataset.image_ids

        train_dataset.image_ids = train_indices
        val_dataset.image_ids = val_indices
        test_dataset.image_ids = all_train_ids[test_indices]

        criterion = nn.BCELoss(reduction='none')
        out_features = 80
    elif args.dataset == 'geode':
        train_dataset = GeodeDataset(train_transform, version='train')
        val_dataset = GeodeDataset(test_transform, version='train')
        test_dataset = GeodeDataset(test_transform, version = 'test')
        out_features = 40 

        train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
        all_train_ids = train_dataset.image_ids
        train_dataset.image_ids = all_train_ids[train_indices]
        val_dataset.image_ids = all_train_ids[val_indices]
        criterion = nn.BCELoss(reduction='none')
    elif args.dataset == 'imagenet':
        train_dataset = ImageNetDataset(train_transform, version='train')
        val_dataset = ImageNetDataset(test_transform, version='train')
        test_dataset = ImageNetDataset(test_transform, version = 'test')
        out_features = 200

        train_indices, val_indices = train_test_split(np.arange(len(train_dataset.image_ids)), test_size=0.2)
        all_train_ids = train_dataset.image_ids
        train_dataset.image_ids = all_train_ids[train_indices]
        val_dataset.image_ids = all_train_ids[val_indices]
        criterion = nn.BCELoss(reduction='none')

    elif args.dataset == 'something':
        pass
        # define train, val, test datasets
        # define out_features
        # define criterion

    if args.toy:
        train_dataset.image_ids = train_dataset.image_ids[:500]
        val_dataset.image_ids = val_dataset.image_ids[-500:]
        test_dataset.image_ids = test_dataset.image_ids[:500]
    elif args.train_num != -1:
        if not dontreduce:
            if args.train_num < 100 and 'celeba' in args.dataset: # likely to only have one label
                len_unique = 0
                while len_unique != 2:
                    random_indices = np.random.choice(np.arange(len(train_dataset.image_ids)), size=args.train_num, replace=False)
                    labels = [train_dataset.__getitem__(ind)[1][0] for ind in random_indices]
                    len_unique = len(np.unique(labels))

                train_dataset.image_ids = train_dataset.image_ids[random_indices]
            else:
                train_num = args.train_num
                if args.train_num > len(train_dataset.image_ids):
                    print("NOT ENOUGH IMAGES -- only has: {0} but wanted {1}".format(len(train_dataset.image_ids), args.train_num))
                    train_num = len(train_dataset.image_ids)
                random_indices = np.random.choice(np.arange(len(train_dataset.image_ids)), size=train_num, replace=False)
                train_dataset.image_ids = train_dataset.image_ids[random_indices]

    print("Number of train images: {0}, val images: {1}, test images: {2}".format(len(train_dataset.image_ids), len(val_dataset.image_ids), len(test_dataset.image_ids)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=workers)

    if args.dollarstreet:
        print("Dollar street test images: {}".format(len(dollarstreet_dataset)))
        return train_loader, val_loader, test_loader, out_features, criterion, dollarstreet_loader
    elif args.fairface:
        print("Fair face test images: {}".format(len(fairface_dataset)))
        return train_loader, val_loader, test_loader, out_features, criterion, fairface_loader
    return train_loader, val_loader, test_loader, out_features, criterion


def get_raytune_ranks(folder_with_results):
    save_name = folder_with_results.split('/')[1]
    bests = [0, 0, 0, 100, 0] # lr, wd, epochs, loss, perf
    trial_to_nums = {}
    for trial in os.listdir(folder_with_results):
        if trial[-4:] != 'json':
            folder_loc = folder_with_results+'/'+trial
            params = json.load(open(folder_loc+'/params.json', 'rb')) 
            epochs = [json.loads(line) for line in open(folder_loc+'/result.json','r')]
            val_losses = [epoch['val_loss'] for epoch in epochs]
            if len(val_losses) == 0:
                trial_to_nums[trial] = [params['lr'], params['wd'], 100, 100, 100]
                print("This trial errored: {}".format(trial))
                continue
            trial_to_nums[trial] = [params['lr'], params['wd'], np.argmin(val_losses), np.amin(val_losses), epochs[np.argmin(val_losses)]['val_perf'], np.argmin(val_losses)]

    sorted_trials = [k for k, v in sorted(trial_to_nums.items(), key=lambda item: item[1][3])]

    # find the rank of the other trial with this kv pair
    best_lr, best_wd = trial_to_nums[sorted_trials[0]][0:2]
    lr2, wd2 = trial_to_nums[sorted_trials[1]][0:2]
    lr3, wd3 = trial_to_nums[sorted_trials[2]][0:2]
    rank = -1
    rank2 = -1
    rank3 = -1
    this_val_loss, this_val_perf = 0, 0
    for t, trial in enumerate(sorted_trials[1:]):
        if trial_to_nums[trial][0] == best_lr and trial_to_nums[trial][1] == best_wd:
            if trial_to_nums[trial][2] == 100:
                rank = -1
            else:
                rank = t+1
                this_val_loss = trial_to_nums[trial][3]
                this_val_perf = trial_to_nums[trial][4]
        elif trial_to_nums[trial][0] == lr2 and trial_to_nums[trial][1] == wd2 and ((t+1)!=1):
            if trial_to_nums[trial][2] == 100:
                rank2 = -1
            else:
                rank2 = t+1
        elif trial_to_nums[trial][0] == lr3 and trial_to_nums[trial][1] == wd3 and ((t+1)!=2):
            if trial_to_nums[trial][2] == 100:
                rank3 = -1
            else:
                rank3 = t+1
    return rank, this_val_loss, this_val_perf, [lr2, wd2, rank2, trial_to_nums[sorted_trials[1]][5]], [lr3, wd3, rank3, trial_to_nums[sorted_trials[2]][5]]

def get_model(args, out_features):
    device = args.device
    if 'pretrain' in args.load_weights:
        if 'openimages' in args.load_weights:
            actual_out_features = out_features
            if args.load_weights[args.load_weights.index('open'):args.load_weights.index('_super')] in OI_BIN_OUTPUTS:
                out_features = 1
            else:
                out_features = 600
        elif 'celeba' in args.load_weights or 'fairface' in args.load_weights:
            actual_out_features, out_features = 1, 1
            if 'Rotate' in args.load_weights:
                out_features = 4
        elif 'coco_gender' in args.load_weights:
            actual_out_features, out_features = 1, 1
        else:
            raise NotImplementedError
    elif args.load_weights[:30] == 'modeloresnet50_weightsoscratch':
        if 'geode' in args.load_weights:
            actual_out_features = out_features
            out_features = 40
        elif 'imagenet' in args.load_weights:
            actual_out_features = out_features
            out_features = 200
        else:
            assert False

    if args.model_type == 'vgg16':
        raise NotImplementedError
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, out_features)
        model.classifier.add_module('last_sigmoid', nn.Sigmoid())
    elif args.model_type == 'alexnet':
        if args.load_weights != 'scratch':
            raise NotImplementedError
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(4096, out_features)
        model.classifier.add_module('last_sigmoid', nn.Sigmoid())
    elif args.model_type == 'resnet101':
        if 'pytorch' in args.load_weights and 'imagenet' in args.load_weights:
            model = models.resnet101(weights=models.resnet.ResNet101_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet101(weights=None)
        model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
    elif args.model_type == 'resnet18':
        if 'pytorch' in args.load_weights and 'imagenet' in args.load_weights:
            model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Linear(512, out_features), nn.Sigmoid())
    elif args.model_type == 'resnet34':
        if 'pytorch' in args.load_weights and 'imagenet' in args.load_weights:
            model = models.resnet34(weights=models.resnet.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34(weights=None)
        model.fc = nn.Sequential(nn.Linear(512, out_features), nn.Sigmoid())
    elif args.model_type == 'resnet50':
        if args.load_weights == 'pytorch_imagenet1k_v1':
            model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
        elif args.load_weights == 'pytorch_imagenet1k_v2' or ('pytorch' in args.load_weights and 'imagenet1k' in args.load_weights):
            model = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
    elif args.model_type == 'vit':
        if 'pytorch' in args.load_weights and 'imagenet' in args.load_weights:
            model = models.vit_b_16(weights=models.vit_b_16.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)
        model.heads.head = nn.Sequential(nn.Linear(768, out_features), nn.Sigmoid())
    elif args.model_type == 'swin':
        if 'pytorch' in args.load_weights and 'imagenet' in args.load_weights:
            model = models.swin_s(weights=models.swin_s.Swin_S_Weights.IMAGENET1K_V1) 
        else:
            model = models.swin_s(weights=None) # TODO: do I want to do other sizes of swin?
        model.head = nn.Sequential(nn.Linear(768, out_features), nn.Sigmoid())

    if 'pretrain' in args.load_weights:

        if 'fairface' in args.load_weights and 'meancalibrate' in args.load_weights and not args.test: # special case where we take the best fair face base
            best_bases = pickle.load(open('fairface_bestbases.pkl', 'rb'))
            att = '_'.join(args.dataset.split('_')[1:])
            assert args.dataset.split('_')[0] == 'celeba'
            setnum = int(args.load_weights[args.load_weights.index('set'):args.load_weights.index('supervision')][3:])
            args.load_weights = args.load_weights.replace('3', best_bases['{0}-{1}'.format(att, setnum)])

        model.load_state_dict(torch.load("models/{0}/start.pt".format(args.load_weights)))
        model.fc = nn.Sequential(nn.Linear(2048, actual_out_features), nn.Sigmoid())
    elif 'random' in args.load_weights:
        path_to_file = 'models/{}/start.pt'.format(args.model_name[:-6])
        if os.path.exists(path_to_file):
            model.load_state_dict(torch.load(path_to_file))
        else:
            if not os.path.exists(path_to_file[:-9]):
                os.makedirs(path_to_file[:-9])
            torch.save(model.state_dict(),path_to_file) 
    elif args.load_weights == 'places':
        model.fc = nn.Sequential(nn.Linear(2048, 365), nn.Softmax())
        assert args.model_type == 'resnet50'
        checkpoint = torch.load('resnet50_places365.pth.tar', map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        keys = list(state_dict.keys())
        for k in keys:
            if k == 'fc.weight':
                state_dict['fc.0.weight'] = state_dict['fc.weight']
                del state_dict['fc.weight']
                state_dict['fc.0.bias'] = state_dict['fc.bias']
                del state_dict['fc.bias']
        model.load_state_dict(state_dict)
        model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
    elif args.load_weights == 'simclr':
        # from https://github.com/Spijkervet/SimCLR
        model.fc = nn.Sequential(nn.Linear(2048, 2048), nn.Softmax())
        assert args.model_type == 'resnet50'
        checkpoint = torch.load('resnet50_simclr.tar', map_location=lambda storage, loc: storage)
        #state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        state_dict = checkpoint
        keys = list(state_dict.keys())
        for k in keys:
            if k[:7] == 'encoder':
                state_dict[k[8:]] = state_dict[k]
                del state_dict[k]
            if k[:9] == 'projector':
                state_dict['fc'+k[9:]] = state_dict[k]
                del state_dict[k]
        del state_dict['fc.2.weight']
        #model.load_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
    elif args.load_weights == 'moco':
        # from https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
        model.fc = nn.Sequential(nn.Linear(2048, 4096), nn.Softmax())
        assert args.model_type == 'resnet50'
        checkpoint = torch.load('resnet50_moco.pth.tar', map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for k in keys:
            if k[:19] == 'module.base_encoder':
                state_dict[k[20:]] = state_dict[k]
                del state_dict[k]
            if 'momentum_encoder' in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.fc = nn.Sequential(nn.Linear(2048, out_features), nn.Sigmoid())
    elif args.load_weights == 'scratch' or 'pytorch' in args.load_weights:
        pass
    elif args.load_weights[:30] == 'modeloresnet50_weightsoscratch':
        model.load_state_dict(torch.load("models/{0}/start.pt".format(args.load_weights)))
        model.fc = nn.Sequential(nn.Linear(2048, actual_out_features), nn.Sigmoid())
    else:
        assert False


    if args.freeze>0:
        for ct, child in enumerate(model.children()): # resnet50 has 10 children
            print("another child")
            if ct < args.freeze:
                for param in child.parameters():
                    param.requires_grad = False
    else:
        assert args.freeze == 0, 'Not implemented more frozen layers'


    model.to(device)

    return model

def init_loss_info(args):
    loss_info = {}
    loss_info['train_labels'] = []
    loss_info['train_probs'] = []
    loss_info['train_epoch'] = []
    loss_info['train_loss'] = []
    loss_info['train_perf'] = []

    loss_info['val_labels'] = []
    loss_info['val_probs'] = []
    loss_info['val_epoch'] = []
    loss_info['val_loss'] = []
    loss_info['val_perf'] = []

    loss_info['test_labels'] = []
    loss_info['test_probs'] = []
    loss_info['test_epoch'] = []

    if 'coco' in args.dataset:
        loss_info['test_gender'] = []
        loss_info['test_skin'] = []
        if args.dollarstreet:
            loss_info['ds_labels'] = []
            loss_info['ds_country'] = []
            loss_info['ds_region'] = []
            loss_info['ds_income'] = []
            loss_info['ds_probs'] = []
            loss_info['ds_epoch'] = []
    if 'openimages' in args.dataset:
        loss_info['test_gender'] = []
    if 'celeba' in args.dataset:
        loss_info['test_gender'] = []
        loss_info['test_young'] = []
        if args.fairface:
            loss_info['ff_probs'] = []
            loss_info['ff_epoch'] = []
            loss_info['ff_labels'] = []
            loss_info['ff_loss'] = []
            loss_info['ff_perf'] = []
    if 'dollarstreet' in args.dataset:
        loss_info['ds_labels'] = []
        loss_info['ds_country'] = []
        loss_info['ds_region'] = []
        loss_info['ds_income'] = []
        loss_info['ds_probs'] = []
        loss_info['ds_epoch'] = []
    return loss_info


def train(args, model, device, train_loader, optimizer, epoch, criterion, loss_info, writer):
    model.train()
    all_probs = []
    all_labels = []
    
    running_loss = []
    for batch_idx, (data, target) in (enumerate(tqdm(train_loader)) if args.interact else enumerate(train_loader)):
        target = target.float()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        
        all_labels.extend(target.data.cpu().numpy())
        all_probs.extend(output.data.cpu().numpy())

        if 'openimages' in args.dataset:
            if train_loader.dataset.has_machine:
                target[target>=.5] = 1
                target[target==-1] = 0
                loss = criterion(output, target).mean()
            else:
                multiplier = torch.absolute(target)
                target[target==-1] = 0
                loss = criterion(output, target)
                loss = torch.multiply(multiplier, loss)
                if multiplier.sum() == 0:
                    print("This should never happen")
                    loss = 0
                else:
                    loss = (loss.sum()) / (multiplier.sum())
        else:
            loss = criterion(output, target)
            loss = loss.mean()
        loss.backward()
        
        optimizer.step()
        running_loss.append(loss.item()*len(data))

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    all_preds = all_probs > .5
    corrects = np.equal(all_preds, all_labels)
    running_batch_loss = np.sum(running_loss) / len(all_labels)
    if len(all_preds.shape) == 1:
        all_preds = np.expand_dims(all_preds, 1)

    if epoch == 1:
        loss_info['train_labels'].append(all_labels)
    #if args.recordall:
    #    loss_info['train_probs'].append(all_probs)

    if 'coco' in args.dataset:
        obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
        perf_score = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])
    elif 'openimages' in args.dataset:
        if train_loader.dataset.has_machine:
            all_labels[all_labels>=.5] = 1
            all_labels[all_labels==-1] = 0
            obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
            perf_score = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])

        else:
            perf_score = partial_average_precision_score(all_labels, all_probs)

    elif args.dataset == 'oi_gender':
        scores = []
        for c in range(len(all_labels[0])):
            scores.append(roc_auc_score(all_labels[:, c], all_probs[:, c]))
        #perf_score = average_precision_score(all_labels, all_probs)
        perf_score = np.mean(scores)
    elif 'celeba' in args.dataset or 'fairface' in args.dataset:
        if len(np.unique(all_labels[:, 0])) > 2:
            perf_score = r2_score(all_labels[:, 0], all_probs[:, 0])
        else:
            perf_score = roc_auc_score(all_labels[:, 0], all_probs[:, 0])
    elif 'dollarstreet' in args.dataset:
        obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
        perf_score = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])
    elif args.dataset in ['geode', 'imagenet']:
        obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
        perf_score = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])


    writer.add_scalar('Loss/train', running_batch_loss, epoch)
    writer.add_scalar('Perf/train', perf_score, epoch)
    loss_info['train_loss'].append(running_batch_loss)
    loss_info['train_perf'].append(perf_score)


    return loss_info

# different from test because does not expect sensitive attribute
def val(args, model, device, test_loader, epoch, criterion, loss_info, writer, label='val'):
    model.eval()
    test_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in (enumerate(tqdm(test_loader)) if args.interact else enumerate(test_loader)):
            target = target.float()
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            loss = criterion(output, target).mean() # sum up batch loss
            test_loss += (loss.item() * len(target))

            all_labels.extend(target.data.cpu().numpy())
            all_probs.extend(output.data.cpu().numpy())

    test_loss /= len(all_labels)
    all_labels, all_probs =  np.array(all_labels), np.array(all_probs)

    if (epoch == args.epochs or epoch % 10 == 0) or args.recordall:
        loss_info['{}_probs'.format(label)].append(all_probs)
        loss_info['{}_labels'.format(label)].append(all_labels)
        loss_info['{}_epoch'.format(label)].append(epoch)
        # TODO: change based on the dataset
   
    test_perf = 0
    if 'coco' in args.dataset or 'dollarstreet' in args.dataset:
        obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
        if len(obj_present) != 0:
            test_perf = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])
    elif 'openimages' in args.dataset:

        if test_loader.dataset.has_machine:
            all_labels[all_labels>=.5] = 1
            all_labels[all_labels==-1] = 0
            obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
            test_perf = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])
        else:
            test_perf = partial_average_precision_score(all_labels, all_probs)
    elif args.dataset == 'oi_gender':
        scores = []
        for c in range(len(all_labels[0])):
            scores.append(roc_auc_score(all_labels[:, c], all_probs[:, c]))
        test_perf = np.mean(scores)
    elif 'celeba' in args.dataset or 'fairface' in args.dataset:
        if len(np.unique(all_labels[:, 0])) > 2:
            test_perf = r2_score(all_labels[:, 0], all_probs[:, 0])
        else:
            test_perf = roc_auc_score(all_labels[:, 0], all_probs[:, 0])
    elif args.dataset in ['geode', 'imagenet']:
        obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
        test_perf = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])

    writer.add_scalar('Loss/{}'.format(label), test_loss, epoch)
    writer.add_scalar('Perf/{}'.format(label), test_perf, epoch)

    loss_info['{}_loss'.format(label)].append(test_loss)
    loss_info['{}_perf'.format(label)].append(test_perf)

    return loss_info

def test(args, model, device, test_loader, epoch, criterion, loss_info, writer, label='val'):
    model.eval()
    test_loss = 0
    all_probs = []
    all_labels = []
    if 'coco' in args.dataset and label != 'ds':
        all_gender = []
        all_skin = []
    elif 'openimages' in args.dataset:
        all_gender = []
    elif label ==  'ds' or 'dollarstreet' in args.dataset:
        all_country = []
        all_region = []
        all_income = []
    elif 'celeba' in args.dataset:
        all_gender = []
        all_young = []
    elif 'fairface' in args.dataset:
        all_gender = []
        all_race = []

    with torch.no_grad():
        for batch_idx, (data, target, attribute) in (enumerate(tqdm(test_loader)) if args.interact else enumerate(test_loader)):
            target = target.float()
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            if label != 'ds' and 'dollarstreet' not in args.dataset:
                loss = criterion(output, target).mean() # sum up batch loss
                test_loss += (loss.item() * len(target))
            else:
                country, region, income = attribute
                all_country.extend(country)
                all_region.extend(region)
                all_income.extend(income)
            all_labels.extend(target.data.cpu().numpy())
            all_probs.extend(output.data.cpu().numpy())
            if 'coco' in args.dataset and label != 'ds':
                gender, skin = attribute
                all_gender.extend(gender)
                all_skin.extend(skin)
            elif 'openimages' in args.dataset:
                attribute = [att.data.cpu().numpy() for att in attribute]
                attribute = np.array(attribute).T
                all_gender.append(attribute)
            elif 'celeba' in args.dataset:
                young, gender = attribute
                all_young.extend(young.data.cpu().numpy())
                all_gender.extend(gender.data.cpu().numpy())
            elif 'fairface' in args.dataset:
                race, gender = attribute
                all_gender.extend(gender)
                all_race.extend(race)

    test_loss /= len(all_labels)
    all_labels, all_probs =  np.array(all_labels), np.array(all_probs)

    loss_info['{}_labels'.format(label)] = all_labels
   
    test_perf = 0
    if 'coco' in args.dataset and label != 'ds':
        obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
        test_perf = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])
    elif 'openimages' in args.dataset:
        if test_loader.dataset.has_machine:
            all_labels[all_labels>=.5] = 1
            all_labels[all_labels==-1] = 0
            obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
            test_perf = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])
            writer.add_scalar('APS/{}'.format(label), test_perf, epoch)
        else:
            test_perf = partial_average_precision_score(all_labels, all_probs)
            writer.add_scalar('AUC/{}'.format(label), partial_auc_score(all_labels, all_probs), epoch)


    elif args.dataset == 'oi_gender':
        scores = []
        for c in range(len(all_labels[0])):
            scores.append(roc_auc_score(all_labels[:, c], all_probs[:, c]))
        test_perf = np.mean(scores)
    elif label == 'ds' or 'dollarstreet' in args.dataset:
        top5 = [np.argsort(prob)[-5:] for prob in all_probs]
        corrects = [len(set(top5[p])&set(np.where(all_labels[p])[0]))>0 for p in range(len(all_probs))]

        test_perf = np.mean(corrects) 
        print("DS test perf: {}".format(test_perf))
    elif 'celeba' in args.dataset or 'fairface' in args.dataset:
        if len(np.unique(all_labels[:, 0])) > 2:
            test_perf = r2_score(all_labels[:, 0], all_probs[:, 0])
        else:
            test_perf = roc_auc_score(all_labels[:, 0], all_probs[:, 0])
    elif args.dataset in ['geode', 'imagenet']:
        obj_present = np.where(np.sum(all_labels, axis=0)!=0)[0]
        test_perf = average_precision_score(all_labels[:, obj_present], all_probs[:, obj_present])

    writer.add_scalar('Loss/{}'.format(label), test_loss, epoch)
    writer.add_scalar('Perf/{}'.format(label), test_perf, epoch)

    if (epoch == args.epochs or epoch % 10 == 0 or epoch == 1) or args.recordall:
        loss_info['{}_probs'.format(label)].append(all_probs)
        loss_info['{}_epoch'.format(label)].append(epoch)
        if 'coco' in args.dataset:
            if label == 'ds':
                loss_info['ds_country'].append(all_country)
                loss_info['ds_region'].append(all_region)
                loss_info['ds_income'].append(all_income)
            else:
                loss_info['{}_gender'.format(label)].append(all_gender)
                loss_info['{}_skin'.format(label)].append(all_skin)
        elif 'openimages' in args.dataset:
            all_gender = np.array(np.concatenate(all_gender))
            if (epoch == args.epochs or epoch % 10 == 0 or epoch == 1):
                loss_info['{}_gender'.format(label)].append(all_gender)
        elif 'celeba' in args.dataset:
            loss_info['{}_gender'.format(label)].append(all_gender)
            loss_info['{}_young'.format(label)].append(all_young)
        elif 'dollarstreet' in args.dataset:
            loss_info['ds_country'].append(all_country)
            loss_info['ds_region'].append(all_region)
            loss_info['ds_income'].append(all_income)

    return loss_info

def raytune(config, checkpoint_dir=None, data_dir=None):
    args = config['args']
    device = args.device
    train_loader, val_loader, test_loader, out_features, criterion = load_data(args)
    writer = SummaryWriter(log_dir='tensorboard/{0}'.format(args.model_name))  

    model = get_model(args, out_features)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=args.momentum, weight_decay=config['wd'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 if epoch > args.epoch_jump else 1)
    loss_info = init_loss_info(args)

    for epoch in range(1, 1+args.epochs):
        loss_info = train(args, model, device, train_loader, optimizer, epoch, criterion, loss_info, writer)
        loss_info = val(args, model, device, val_loader, epoch, criterion, loss_info, writer, label='val')
        scheduler.step()
        tune.report(val_loss=loss_info['val_loss'][-1], val_perf=loss_info['val_perf'][-1], epoch=epoch, train_loss=loss_info['train_loss'][-1], train_perf=loss_info['train_perf'][-1])

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--epoch_jump', type=int, default=50, help='after how many epochs to change lr by .1')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--hp_search', action='store_true', default=False, help='if set, searching lr and wd')
    parser.add_argument('--hp_num_samples', type=int, default=2, help='number of samples for raytune')
    parser.add_argument('--hp_grace', type=int, default=5, help='raytune grace period of epochs')

    parser.add_argument('--model_type', type=str, default="resnet_50", help='model type')
    parser.add_argument('--load_weights', type=str, default='scratch', help='which weights to load, scratch is random init, random[1-5] is kept random, pretrain_datao{0}_supervisiono{1}.format(args.dataset, args.supervision) is loading the pretrained weights')
    parser.add_argument('--dataset', type=str, default='imagenet', help='has to be [imagenet, places, openimages_v0, openimages_v1, openimages_v2, coco, dollarstreet]')
    parser.add_argument('--dollarstreet', action='store_true', default=False, help='also test on dollarstreet')
    parser.add_argument('--fairface', action='store_true', default=False, help='also test on fairface')
    parser.add_argument('--supervision', type=int, default=0, help='0 is supervisesd, 1+ are going to be different self-supervision') # TODO: implemenet self-supervision stuff
    parser.add_argument('--train_num', type=int, default=-1, help='number of training examples to do, -1 default is all of them')
    parser.add_argument('--duplicate', type=int, default=0, help='which duplicate model it is, will end up having 1-5 for confidence intervals, 0 is for hp search where it doesnt matter')
    parser.add_argument('--save', action='store_true', default=False, help='save the model ckpt')

    parser.add_argument('--hp_trainnum', type=int, default=-2, help='which set of hps to use. -2 means whatever the trainnum is')
    parser.add_argument('--freeze', type=int, default=0, help='number of layers to freeze')

    parser.add_argument('--toy', action='store_true', default=False,
                        help='toy dataset so 500 samples')
    parser.add_argument('--test', action='store_true', default=False,
                        help='do not train only test')
    parser.add_argument('--recordall', action='store_true', default=False,
                        help='record all predictions for train val test etc etc')
    parser.add_argument('--testreg', action='store_true', default=False,
                        help='do not make the test set match whatever is happening')
    parser.add_argument('--manual', action='store_true', default=False,
                        help='override getting optimal hyperparameters, and just do what it is actually given')
    parser.add_argument('--interact', action='store_true', default=False,
                        help='For if showing tqdm')
    parser.add_argument('--extra_name', type=str, default="", help='extra name to tack on the end of args model name')
    args = parser.parse_args()
    print(args)
    use_cuda = torch.cuda.is_available()

    assert args.duplicate >= 0 and args.duplicate < 10
    if args.dollarstreet: ## note that this just tests on dollarstreet when the training is coco -- the dollarstreet argument trains on it as well
        assert 'coco' in args.dataset
        assert not args.hp_search
    if args.fairface:
        assert 'celeba' in args.dataset
        assert not args.hp_search

    if args.hp_search:
        assert args.duplicate == 0
    if args.hp_trainnum == -2:
        args.hp_trainnum = args.train_num
    
    load_weights_name = args.load_weights
    if '/' in args.load_weights:
        load_weights_name = args.load_weights.replace('/', '7')
    args.model_name = 'modelo{0}_weightso{1}_datao{2}_traino{3}_dupo{4}'.format(args.model_type, load_weights_name, args.dataset, args.train_num, args.duplicate)
    if args.extra_name != '':
        args.model_name += '_{}'.format(args.extra_name)
        assert args.duplicate == 0

    print("Model name: {}".format(args.model_name))
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    print("Cuda devices number: {}".format(torch.cuda.device_count()))
    cuda_name = 'cuda'
    device = torch.device(cuda_name if use_cuda else "cpu")
    print("Device: {}".format(device))
    args.device = device

    if not args.hp_search and not args.test:
        if not args.manual:
            hpresult_loc = re.sub(r'dupo[1-9]', 'dupo0', args.model_name)
            if 'traino-17' in hpresult_loc:
                hpresult_loc = hpresult_loc.replace('traino-17dupo0', 'traino-17dupo1')
            hpresult_loc = re.sub(r'random[1-5]', 'scratch', hpresult_loc)
            if args.extra_name != '':
                hpresult_loc = hpresult_loc.split('_')[:-1]
                hpresult_loc = '_'.join(hpresult_loc)

            hpresult_loc = re.sub(r'traino[0-9]+', 'traino{}'.format(args.hp_trainnum), hpresult_loc)
            try:
                load_results = pickle.load(open('hpsearch_results/{}.pkl'.format(hpresult_loc), 'rb'))
            except:
                print("HP Result Loc looking somewhere else, not found: {}".format(hpresult_loc))
                if 'modeloresnet50_weightsopretrain_dataofairface' == hpresult_loc[:45] and hpresult_loc[hpresult_loc.index('dataoceleba'):hpresult_loc.index('dupo0')].count('+') == 2:
                    load_results = pickle.load(open('hpsearch_results/{}.pkl'.format(re.sub(r'set[0-9]', 'set5', hpresult_loc)), 'rb'))
                elif 'coco_dsset' in args.dataset:
                    new_loc = re.sub(r'set[0-9]', 'set5', hpresult_loc.replace('set10', 'set5'))
                    load_results = pickle.load(open('hpsearch_results/{}.pkl'.format(new_loc), 'rb'))

            if len(load_results) == 6:
                args.lr = load_results[0]
                args.wd = load_results[1]
                args.epochs = load_results[2]+1
                loss, perf = load_results[3], load_results[4]
                folder_with_results = load_results[5]
                print("LR {0}, WD {1}, Epoch {2} has val loss {3:.4f} and perf {4:.4f}".format(load_results[0], load_results[1], load_results[2]+1, load_results[3], load_results[4]))

                rank, this_val_loss, this_val_perf, rank2, rank3 = get_raytune_ranks(folder_with_results)
            else:
                bt_config, bt_trialid, bt_metricanalysis, bt_localdir = load_results

                args.lr = bt_config['lr']
                args.wd = bt_config['wd']

                # to find out best epochs
                folder_loc = bt_localdir + '/'
                found = False
                rank, this_val_loss, this_val_perf, rank2, rank3 = get_raytune_ranks(folder_loc[:-1])
                for trial in os.listdir(bt_localdir):
                    if bt_trialid in trial:
                        assert not found, "Duplicates for best trial name"
                        folder_loc += trial
                        found = True
               
                folder_loc += '/result.json'

                epochs = [json.loads(line) for line in open(folder_loc,'r')]
                val_losses = [epoch['val_loss'] for epoch in epochs]
                best_epoch = np.argmin(val_losses)
                print("LR {0}, WD {1}, Epoch {2} has val loss {3:.4f} and perf {4:.4f}".format(args.lr, args.wd, best_epoch+1, epochs[best_epoch]['val_loss'], epochs[best_epoch]['val_perf']))
                args.epochs = best_epoch+1
            print("Rank of other run is {0} with loss {1:.4f} and perf {2:.4f}".format(rank, this_val_loss, this_val_perf))
            if rank <= 5: ## doing something different, potentially picking the 2nd or 3rd best
                if rank2[2] <= 5:
                    to_use = 1
                elif rank3[2] <= 5:
                    to_use = 2
                else:
                    to_use = np.argmin([rank, rank2[2]-1, rank3[2]-2])

                if to_use == 1:
                    args.lr = rank2[0]
                    args.wd = rank2[1]
                    args.epochs = rank2[3]+1
                    lowest = rank2[2]
                elif to_use == 2:
                    args.lr = rank3[0]
                    args.wd = rank3[1]
                    args.epochs = rank3[3]+1
                    lowest = rank3[2]
                else:
                    lowest = rank
                print("Instead using LR {0}, WD {1}, Epoch {2} -- lowest rank is {3}".format(args.lr, args.wd, args.epochs, lowest))
        writer = SummaryWriter(log_dir='tensorboard/{}'.format(args.model_name)) 

    if args.hp_search:
        assert 'random' not in args.load_weights, 'just say scratch not random for load_weights when doing hp search'
        ray.init(num_cpus=16, num_gpus=4)
        config = {
                "lr": tune.grid_search([.1, .05, .01, .005, .001]),
                "wd": tune.grid_search([.0, .0001, .0005, .001]),
                'args': args
            }
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=args.epochs,
            grace_period=args.hp_grace,
            reduction_factor=2)

        reporter = CLIReporter(
            metric_columns=["val_loss", "val_perf", "epoch", 'train_loss', 'train_perf'])

        result = tune.run(
            partial(raytune),
            resources_per_trial={"cpu": 4, "gpu": 1},
            config=config,
            num_samples=args.hp_num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir='raytune_results/{}'.format(args.model_name),
            verbose=1)

        best_trial = result.get_best_trial("val_loss", "min", "all") # instead of 'last', maybe do 'all' or something??
        pickle.dump([best_trial.config, best_trial.trial_id, best_trial.metric_analysis, best_trial.local_dir], open('hpsearch_results/{}.pkl'.format(args.model_name), 'wb'))
        print("Best trial config: {}".format(best_trial.config))
        
        exit()

    if args.dollarstreet:
        train_loader, val_loader, test_loader, out_features, criterion, dollarstreet_loader = load_data(args)
    elif args.fairface:
        train_loader, val_loader, test_loader, out_features, criterion, fairface_loader = load_data(args)
    else:
        train_loader, val_loader, test_loader, out_features, criterion = load_data(args)

    model = get_model(args, out_features)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1 if epoch > args.epoch_jump else 1) # TODO: change to right step which depends on total epochs
    loss_info = init_loss_info(args)

    if args.testreg:
        loss_info['test_reg'] = True

    start = time.time()

    #folder = "models/" + args.model_name
    #if not os.path.exists(folder):
    #    os.makedirs(folder)
    if args.test:
        writer = SummaryWriter(log_dir='tensorboard/{}'.format(args.model_name)) 
        loss_info = val(args, model, device, val_loader, 1, criterion, loss_info, writer, label='val')
        loss_info = test(args, model, device, test_loader, 1, criterion, loss_info, writer, label='test')
        if args.dollarstreet:
            loss_info = test(args, model, device, dollarstreet_loader, 1, criterion, loss_info, writer, label='ds')
        elif args.fairface:
            loss_info = val(args, model, device, fairface_loader, 1, criterion, loss_info, writer, label='ff')
    else:
        for epoch in range(1, args.epochs + 1):
            print("Epoch: {}".format(epoch))
            loss_info = train(args, model, device, train_loader, optimizer, epoch, criterion, loss_info, writer)
            if epoch == args.epochs:
                loss_info = val(args, model, device, val_loader, epoch, criterion, loss_info, writer, label='val') # only need val for hp search, except last one I want to know scores
            loss_info = test(args, model, device, test_loader, epoch, criterion, loss_info, writer, label='test')
            if (epoch % 5 == 0) or ((epoch+5) > args.epochs):
                if args.dollarstreet:
                    loss_info = test(args, model, device, dollarstreet_loader, epoch, criterion, loss_info, writer, label='ds')
                elif args.fairface and epoch == args.epochs:
                    loss_info = val(args, model, device, fairface_loader, epoch, criterion, loss_info, writer, label='ff')

            scheduler.step()

    end = time.time()
    print("\nTook {:.2f} minutes".format((end-start)/60.))

    writer.close()

    if args.save:
        if args.extra_name != "":
            folder = "models/{0}".format(args.model_name)
        else:
            folder = "models/{0}/{1}".format(args.model_name[:-6], args.model_name[-5:])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(loss_info, open("{0}/loss_info.pkl".format(folder), 'wb')) 
        torch.save(model.state_dict(),"{}/start.pt".format(folder)) 

    
    if (('random' in args.load_weights or 'scratch' in args.load_weights) and 'coco' not in args.dataset): # must be pretraining then
        if args.duplicate != 0:
            args.extra_name = args.duplicate

        save_name = 'pretrain_datao{0}_supervisiono{1}_traino{2}_ejo{3}'.format(args.dataset, args.supervision, args.train_num, args.epoch_jump)
        if args.extra_name != '':
            save_name += '_{}'.format(args.extra_name)
        folder = "models/{0}".format(save_name)
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        if args.train_num != -1 and 'celeba' in args.dataset:
            pass
        else:
            torch.save(model.state_dict(),"{}/start.pt".format(folder)) 
        pickle.dump(loss_info, open("{0}/loss_info.pkl".format(folder), 'wb')) 
    elif ('imagenet' in args.load_weights and args.dataset in ['openimages_v500kw14_13', 'openimages_v500kw14_9']) or ('imagenet' in args.load_weights and args.dataset in ['celeba_Male', 'celeba_Bangs', 'celeba_Wearing_Hat', 'celeba_Rotate'] and args.manual):## specific case of imagenet finetuned on oi v9 or v13 is STILL going to be a pretrained model
        save_name = 'pretrain_datao{0}_supervisiono{1}_traino{2}_ejo{3}'.format('pretrainimagenet'+args.dataset, args.supervision, args.train_num, args.epoch_jump)
        if args.extra_name != '':
            save_name += '_{}'.format(args.extra_name)
        folder = "models/{0}".format(save_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(model.state_dict(),"{}/start.pt".format(folder)) 
        pickle.dump(loss_info, open("{0}/loss_info.pkl".format(folder), 'wb')) 
    else: # a from scratch or finetuning model
        if args.extra_name != "":
            folder = "models/{0}".format(args.model_name)
        else:
            folder = "models/{0}/{1}".format(args.model_name[:-6], args.model_name[-5:])
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(loss_info, open("{0}/loss_info.pkl".format(folder), 'wb')) 
    print("Last val perf: {0:.4f}".format(loss_info['val_perf'][-1]))
        
if __name__ == '__main__':
    main()


