import pickle
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import os
import pandas as pd
import argparse
import sys
sys.path.append('..')
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=int, default=0, help='0 is relative to gender, 1 is attributes relative to each other')
args = parser.parse_args()

annotations = pd.read_csv('/Data/CelebA/celeba/list_attr_celeba.txt', delim_whitespace=True, header=1)

attributes_11 = ['Mouth_Slightly_Open', 'Smiling', 'Eyeglasses', 'Black_Hair', 'Narrow_Eyes', 'Gray_Hair', 'High_Cheekbones', 'Wearing_Earrings', 'Brown_Hair', 'Blond_Hair', 'Bags_Under_Eyes']
bases_3 = ['Wearing_Hat', 'Bangs', 'Chubby']

if args.version == 0:
    categories = ATTRIBUTES_11 + BASES_3
    has_categories = []
    saliencies = []
    skews = []
    for cat in categories:
        name_both = 'celeba_{}&Male'.format(cat)
        name_inverse = 'celeba_{}&Male&'.format(cat)
        cat_scores = []
        gen_scores = []

        for name in [name_both, name_inverse]:
            loss_info = pickle.load(open('models/pretrain_datao{}_supervisiono0_traino-1_ejo50/loss_info.pkl'.format(name), 'rb'))
            this_cat_score = []
            this_gen_score = []
            for l in ['a', 'b', 'c', 'd', 'e']:
                loss_info = pickle.load(open('models/pretrain_datao{0}_supervisiono0_traino-1_ejo50_{1}/loss_info.pkl'.format(name, l), 'rb'))
                probs = np.array(loss_info['test_probs'][-1])
                labels = loss_info['test_labels']
                cat_labs = np.array(loss_info['test_young'][-1])
                gen_labs = np.array(loss_info['test_gender'][-1])
                cat_score = np.amax([roc_auc_score(cat_labs, probs), roc_auc_score(1-cat_labs, probs)])
                gen_score = np.amax([roc_auc_score(gen_labs, probs), roc_auc_score(1-gen_labs, probs)])
                this_cat_score.append(cat_score)
                this_gen_score.append(gen_score)
            cat_scores.append(np.mean(this_cat_score))
            gen_scores.append(np.mean(this_gen_score))

        cat_scores, gen_scores =  np.array(cat_scores), np.array(gen_scores)
        saliency = np.mean(cat_scores)/np.mean(gen_scores)
        saliency = np.mean(gen_scores)-np.mean(cat_scores)+.5
        saliencies.append(saliency)
        has_categories.append(cat)
        has_att = np.where(np.array(annotations[cat]) == 1)[0]
        fem = np.where(np.array(annotations['Male'])==-1)[0]
        masc = np.where(np.array(annotations['Male'])==1)[0]
        skews.append(len(set(fem)&set(has_att))/len(has_att))

    fem_num = len(fem)/len(annotations)
    order = np.argsort(saliencies)
    att_to_s = {}
    for o in order:
        att_to_s[has_categories[o]] = [saliencies[o], skews[o]]

    pickle.dump([fem_num, att_to_s], open('att_to_s.pkl', 'wb'))
elif args.version == 1:
    saliences = {}

    for pair in ATTRIBUTE_PAIRS:
        lab1, lab2 = pair
        name_both = 'celeba_{0}&{1}'.format(lab1, lab2)
        name_inverse = 'celeba_{0}&{1}&'.format(lab1, lab2)
        rev_name = False
        if (not os.path.exists('models/pretrain_datao{}_supervisiono0_traino-1_ejo10_a/loss_info.pkl'.format(name_both))) or (not os.path.exists('models/pretrain_datao{}_supervisiono0_traino-1_ejo10_a/loss_info.pkl'.format(name_inverse))):
            rev_name = True
            name_both = 'celeba_{0}&{1}'.format(lab2, lab1)
            name_inverse = 'celeba_{0}&{1}&'.format(lab2, lab1)
        
        cat_scores = []
        gen_scores = []
        diff_scores = []
        for name in [name_both, name_inverse]:
            this_cat_score = []
            this_gen_score = []
            this_diff_score = []
            overall_auc = []
            for l in ['a', 'b', 'c', 'd', 'e']:
                loss_info = pickle.load(open('models/pretrain_datao{0}_supervisiono0_traino-1_ejo50_{1}/loss_info.pkl'.format(name, l), 'rb'))
                probs = np.array(loss_info['test_probs'][-1])
                labels = np.array(loss_info['test_labels'])
                overall_auc.append(roc_auc_score(labels, probs))
                cat_labs = np.array(loss_info['test_young'][-1])
                gen_labs = np.array(loss_info['test_gender'][-1])
                cat_score = np.amax([roc_auc_score(cat_labs, probs), roc_auc_score(1-cat_labs, probs)])
                gen_score = np.amax([roc_auc_score(gen_labs, probs), roc_auc_score(1-gen_labs, probs)])
                this_cat_score.append(cat_score)
                this_gen_score.append(gen_score)
                this_diff_score.append(gen_score-cat_score)
            cat_scores.append(np.mean(this_cat_score))
            gen_scores.append(np.mean(this_gen_score))
            diff_scores.append(np.mean(this_diff_score))
        cat_scores, gen_scores =  np.array(cat_scores), np.array(gen_scores)
        saliency = np.mean(gen_scores) - np.mean(cat_scores)
        cat = '{0}-{1}'.format(lab1, lab2)
        if rev_name:
            cat = '{0}-{1}'.format(lab2, lab1)
        saliences[cat] = saliency
    pickle.dump([saliences], open('celeba/geo_sals.pkl', 'wb'))

