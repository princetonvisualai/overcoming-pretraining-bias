import argparse
import os
from utils import *

def classifier(keyword_dict):
    command = 'python3 classifier.py'
    for key in keyword_dict.keys():
        if type(keyword_dict[key]) == str and len(keyword_dict[key]) > 0:
            command += ' --{0} "{1}"'.format(key, keyword_dict[key])
        else:
            command += ' --{0} {1}'.format(key, keyword_dict[key])
    os.system(command)

def main():
    parser = argparse.ArgumentParser(description='Which experiments to replicate')
    parser.add_argument('--bias_type', type=int, default=0, help='0 is bias as spurious correlations, 1 is bias as underrepresentation')
    parser.add_argument('--type', nargs='+', type=int, default=[0, 1], help='0 is train all of the models, 1 is run analysis once models are trained')
    parser.add_argument('--experiments', nargs='+', type=int, default=[0, 1], help='0 is experiments showing finetuned models inherit pretrained bias, 1 is experiments showing how finetuned datasets can correct for pretrained bias')
    parser.add_argument('--datasets', action='store_true', default=False, help='sets up dataset files, needs to be run the first time to generate certain annotations') 
    args = parser.parse_args()

    assert len(set(args.experiments).difference(set([0, 1]))) == 0, "Experiments must only include 0 or 1"
    assert len(set(args.type).difference(set([0, 1]))) == 0, "Type must only include 0 or 1"
    assert args.bias_type in [0, 1], "Bias type can only be 0 or 1"

    if args.datasets:
        os.system('python3 dataset_files/coco_gender.py') # derive gender labels on coco from the captions
        os.system('python3 dataset_files/pick_objects.py') # generate the splits on coco objects with different correlation strengths
        os.system('python3 dataset_files/celeba_splits.py') # generate the splits on celeba attributes with different correlation strengths

    if args.bias_type == 0:
        if 0 in args.type:
            if 0 in args.experiments: # train models

                # (1) Comparing relative salience of attributes to gender
                for attribute in (ATTRIBUTES_11 + BASES_3):
                    classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}&Male'.format(attribute), 'hp_search': ''})
                    classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}&Male&'.format(attribute), 'hp_search': ''})
                    for d in ['a', 'b', 'c', 'd', 'e']:
                        classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}&Male'.format(attribute), 'extra_name': d})
                        classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}&Male&'.format(attribute), 'extra_name': d})

                # (2) Getting Gendered and Control bases
                for attribute in (['Male'] + BASES_3):
                    classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}'.format(attribute), 'hp_search': ''})
                    classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}'.format(attribute)})

                # (3) Training 11 attributes on the 4 bases
                for pretrain in (['Male'] + BASES_3):
                    for finetune in ATTRIBUTES_11:
                        classifier({'load_weights': 'pretrain_dataoceleba_{}_supervisiono0_traino-1_ej50'.format(pretrain), 'dataset': 'celeba_{}'.format(finetune), "train_num": 1024, 'hp_search': ''}) # correlation strength is natural distribution
                        classifier({'load_weights': 'pretrain_dataoceleba_{}_supervisiono0_traino-1_ej50'.format(pretrain), 'dataset': 'celeba_{}_skewz'.format(finetune), "train_num": 1024, 'hp_search': ''}) # correlation strength artificially set to be .8
                        for d in np.arange(1, 6):
                            classifier({'load_weights': 'pretrain_dataoceleba_{}_supervisiono0_traino-1_ej50'.format(pretrain), 'dataset': 'celeba_{}'.format(finetune), "train_num": 1024, 'duplicate': d}) # correlation strength is natural distribution
                            classifier({'load_weights': 'pretrain_dataoceleba_{}_supervisiono0_traino-1_ej50'.format(pretrain), 'dataset': 'celeba_{}_skewz'.format(finetune), "train_num": 1024, 'duplicate': d}) # correlation strength artificially set to be .8

                # (4) For the 4 biased attributes, manipulating the finetuning number
                for finetune in BIAS_ATTRIBUTES_4:
                    for n in [64, -1]:
                        for pretrain in (['Male'] + BASES_3):
                            classifier({'load_weights': 'pretrain_dataoceleba_{}_supervisiono0_traino-1_ej50'.format(pretrain), 'dataset': 'celeba_{}'.format(finetune), "train_num": n, 'hp_search': ''}) 
                        classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}'.format(finetune), "train_num": n, 'hp_search': ''}) 

                    for n in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, -1]:
                        if n <= 128:
                            hp_num = 64
                        else:
                            hp_num = -1
                        for d in np.arange(1, 6):
                            for pretrain in (['Male'] + BASES_3):

                                classifier({'load_weights': 'pretrain_dataoceleba_{}_supervisiono0_traino-1_ej50'.format(pretrain), 'dataset': 'celeba_{}'.format(finetune), "train_num": n, 'hp_trainnum': hp_num, 'duplicate': d}) 
                            classifier({'load_weights': 'scratch', 'dataset': 'celeba_{}'.format(finetune), "train_num": n, 'hp_trainnum': hp_num, 'duplicate': d}) 
            if 1 in args.experiments: # analyze results
                # (1) Calculate the relative salience values
                os.system("python3 analysis_files/saliency.py --version 0")

                # (2) Perform the regression on salience and correlation level (i.e., skew)
                os.system("python3 analysis_files/regress.py")

                # (3) Graphing the progression as finetuning number changes
                os.system("python3 analysis_files/trainnum_xaxis.py")
        if 1 in args.type:
            if 0 in args.experiments: # train models
                # (1) Existing pretrained models on 4 CelebA attributes of varying correlation strength while maintaining number of finetuning images
                for pretrain in PRETRAINED_5:
                    for finetune in BIAS_ATTRIBUTES_4:
                        for n in [128, 1024]:
                            classifier({'load_weights': pretrain, 'dataset': 'celeba_{0}_set{1}'.format(finetune, 5), 'train_num': n, 'hp_search': ''})
                            for s in np.arange(0, 11):
                                for d in np.arange(1, 6):
                                    classifier({'load_weights': pretrain, 'dataset': 'celeba_{0}_set{1}'.format(finetune, s), 'train_num': n, 'duplicate': d})

                # (2) Existing pretrained models on 4 CelebA attributes of varying correlation strength while subselecting from data
                for pretrain in PRETRAINED_5:
                    for finetune in ['Wearing_Earrings', 'Blond_Hair']:
                        for s in np.arange(1, 10):
                            classifier({'load_weights': pretrain, 'dataset': 'celeba_{0}_set{1}'.format(finetune, s), 'train_num': constrain_nums2[s], 'hp_search': ''})
                            for d in np.arange(1, 6):
                                classifier({'load_weights': pretrain, 'dataset': 'celeba_{0}_set{1}'.format(finetune, s), 'train_num': constrain_nums2[s], 'duplicate': d})
                    for finetune in ['Brown_Hair', 'Bags_Under_Eyes']:
                        for s in np.arange(3, 12):
                            classifier({'load_weights': pretrain, 'dataset': 'celeba_{0}_set{1}'.format(finetune, s), 'train_num': constrain_nums8[s], 'hp_search': ''})
                            for d in np.arange(1, 6):
                                classifier({'load_weights': pretrain, 'dataset': 'celeba_{0}_set{1}'.format(finetune, s), 'train_num': constrain_nums8[s], 'duplicate': d})

                # (3) Existing pretrained models on 4 COCO objects of varying correlation strength while maintaining number of finetuning images
                for pretrain in PRETRAINED_5:
                    for finetune in COCOOBJ_4:
                        for n in [1000, 5000]:
                            classifier({'load_weights': pretrain, 'dataset': 'coco_obj{0}+set{1}'.format(finetune, 5), 'train_num': n, 'hp_search': ''})
                            for s in np.arange(0, 11):
                                for d in np.arange(1, 6):
                                    classifier({'load_weights': pretrain, 'dataset': 'coco_obj{0}+set{1}'.format(finetune, s), 'train_num': n, 'duplicate': d})

            if 0 in args.experiments: # analyze results
                # (1) Show results on 4 CelebA attributes of varying correlation strength while maintaining number of finetuning images
                os.system("python3 analysis_files/vary_corr.py --version 0 --train_num 128")
                os.system("python3 analysis_files/vary_corr.py --version 0 --train_num 1024")

                # (2) Show results on 4 CelebA attributes of varying correlation strength while subselecting from data
                os.system("python3 analysis_files/vary_corr.py --version 1 --train_num 1024")
                
                # (3) Show results on 4 COCO objects of varying correlation strength while maintaining number of finetuning images
                os.system("python3 analysis_files/vary_corr.py --version 2 --train_num 1000")
                os.system("python3 analysis_files/vary_corr.py --version 2 --train_num 5000")
    if args.bias_type == 1:
        # args.type 0 and 1 is combined for bias as underrepresentation

        if 0 in args.experiments: # train models
            # (1) Find the salience of the 12 pairs we will be using
            for pair in ATTRIBUTE_PAIRS:
                classifier({'load_weights': 'scratch', 'dataset': 'celeba_{0}&{1}'.format(pair[0], pair[1]), 'hp_search': ''})
                classifier({'load_weights': 'scratch', 'dataset': 'celeba_{0}&{1}&'.format(pair[0], pair[1]), 'hp_search': ''})
                for d in ['a', 'b', 'c', 'd', 'e']:
                    classifier({'load_weights': 'scratch', 'dataset': 'celeba_{0}&{1}'.format(pair[0], pair[1]), 'extra_name': d})
                    classifier({'load_weights': 'scratch', 'dataset': 'celeba_{0}&{1}&'.format(pair[0], pair[1]), 'extra_name': d})

            # (2) Create pretrained bases trained on FairFace
            bases = list(np.unique(np.concatenate([pair.split('-') for pair in (GREATER_PAIRS+LESS_PAIRS+SAME_PAIRS)])))
            for att in bases:
                # labels FairFace with this attribute
                classifier({'load_weights': 'pytorch_imagenet1k_v2', 'dataset': 'celeba_{}'.format(att), 'hp_search': ''})
                classifier({'load_weights': 'pytorch_imagenet1k_v2', 'dataset': 'celeba_{}'.format(att), 'fairface': ''})

                # trains a base classifier on this attribute
                classifier({'load_weights': 'scratch', 'dataset': 'fairface_{}'.format(att), 'hp_search': ''})
                classifier({'load_weights': 'scratch', 'dataset': 'fairface_{}'.format(att)})

            # (3) On the 12 CelebA pairs, trains for proportion split 50/50 on 128 and 1024
            for pair in (GREATER_PAIRS+LESS_PAIRS+SAME_PAIRS):
                for pretrain in pair:
                    for finetune in pair:
                        for n in [128, 1024]:
                            classifier({'load_weights': 'pretrain_dataofairface_{}_supervisiono0_traino-1_ejo50'.format(pretrain), 'dataset': 'celeba_{0}+{1}+set{2}-prop{3}'.format(pretrain, finetune, 5, 5), 'train_num': n, 'hp_search': ''})
                            for s in np.arange(0, 11):
                                for d in np.arange(1, 6):
                                    classifier({'load_weights': 'pretrain_dataofairface_{}_supervisiono0_traino-1_ejo50'.format(pretrain), 'dataset': 'celeba_{0}+{1}+set{2}-prop{3}'.format(pretrain, finetune, s, 5), 'train_num': n, 'duplicate': d})

            # (4) Get pretrained bases of GeoDE and ImageNet
            for base in ['geode', 'imagenet']:
                classifier({'load_weights': 'scratch', 'dataset': base, 'hp_search': '', 'epochs': 50})
                classifier({'load_weights': 'scratch', 'dataset': base})

            # (5) Use the GeoDE and ImageNet bases on varying proportions of Dollar Street and COCO
            for pretrain in ['geode', 'imagenet']:
                classifier({'load_weights': "modeloresnet50_weightsoscratch_datao{}_traino-1/dupo0".format(pretrain), 'dataset': 'coco_dsset5', 'train_num': 128, 'hp_search': ''})
                for finetune in np.arange(0, 11):
                    for d in np.arange(1, 6):
                        classifier({'load_weights': "modeloresnet50_weightsoscratch_datao{}_traino-1/dupo0".format(pretrain), 'dataset': 'coco_dsset{}'.format(finetune), 'train_num': 128, 'duplicate': d, 'dollarstreet': ''})


        if 1 in args.experiments: # analyze results
            # (1) Analyze the salience results of the pairs
            os.system("python3 analysis_files/saliency.py --version 1")

            # (2) Analyze the CelebA pairs
            os.system("python3 analysis_files/celeba_pairs.py")

            # (3) Analyze the [GeoDE x ImageNet] pretrained bases on the combination of Dollar Street and COCO
            os.system("python3 analysis_files/geodiverse_props.py")

if __name__ == '__main__':
    main()

