import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import average_precision_score, confusion_matrix, auc, roc_auc_score
import argparse
from statsmodels.regression.linear_model import OLS
import statsmodels.formula.api as smf
import pandas as pd
import copy
import sys
sys.path.append('..')
from utils import *

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--version', type=int, default=0, help='0 is celeba maintaining constant train num, 1 is celeba reducing train num to vary corr, 2 is coco maintaining constant train num')
parser.add_argument('--train_num', type=int, default=128, help='')
parser.add_argument('--two', action='store_true', default=False, help='if true then only show two objects not four, like for paper figure') 
args = parser.parse_args()

version = args.version
train_num =  args.train_num
pre_sets = PRETRAINED_5 
fine_sets = np.arange(0, 11)

fem_num, att_to_s = pickle.load(open('./att_to_s.pkl', 'rb'))
saved_props = pickle.load(open('coco/obj_spec_props.pkl', 'rb'))

width = .4

if version in [0, 1]:
    attributes = ['Wearing_Earrings', 'Blond_Hair', 'Brown_Hair', 'Bags_Under_Eyes']
    if args.two:
        attriburs = ['Blond_Hair', 'Bags_Under_Eyes']
    assert train_num in [128, 1024]
elif version in [2]:
    attributes = ['dining-table', 'handbag', 'chair', 'cup']
    if args.two:
        attributes = ['dining-table', 'chair']
    assert train_num in [1000, 5000]
dups = 5

preset_name = {'scratch': 'Scratch', 'pytorch_imagenet1k_v2': 'TorchVision', 'moco': 'MoCo', 'simclr': 'SimCLR', 'places': 'Places', 'scratch - constrain': 'Scratch_', 'pytorch_imagenet1k_v2 - constrain': 'TorchVision_', 'moco - constrain': 'MoCo_', 'simclr - constrain': 'SimCLR_', 'places - constrain': 'Places_'}

if args.two:
    f, axes = plt.subplots(len(attributes), 1, figsize=(3.4, 5.7)) # length, then height
else:
    f, axes = plt.subplots(1, len(attributes), figsize=(7.2, 2.8)) # length, then height

for a, att in enumerate(attributes):
    dict_diffs = {}
    dict_f_fprs = {}
    dict_m_fprs = {}
    dict_aucs = {}

    min_val = 1
    max_val = -1
    for pre_ind, pre_set in enumerate(pre_sets):
        if version in [0, 1]:
            if att in ['Wearing_Earrings', 'Blond_Hair']:
                incr = 2
            elif att in ['Bags_Under_Eyes', 'Brown_Hair']:
                incr = 8
            else:
                assert NotImplementedError

        for fine_ind, fine_set in enumerate(fine_sets):
            if version in [2]:
                if 'scratch' in pre_set:
                    if '- constrain' not in pre_set:
                        filename = 'pretrain_dataococo_obj{0}+set{1}_supervisiono0_traino{2}_ejo50_manualdupo3'.format(att, fine_set, train_num)
                    else:
                        filename = 'pretrain_dataococo_obj{0}+set{1}_supervisiono0_traino{2}_ejo50_dupo3'.format(att, fine_set, train_num)
                else:
                    if '- constrain' not in pre_set:
                        filename = 'modeloresnet50_weightso{1}_dataococo_obj{0}+set{2}_traino{3}_dupo0_manualdupo3'.format(att, pre_set, fine_set, train_num)
                    else:
                        filename = 'modeloresnet50_weightso{1}_dataococo_obj{0}+set{2}_traino{3}/dupo3'.format(att, pre_set, fine_set, train_num)
            else:
                fine_num = fine_set+1 if fine_set != 10 else 0 ### fine_set 11 and 0 are the same due to +1
                if 'scratch' in pre_set:
                    if '- constrain' in pre_set:
                        try:
                            filename = 'pretrain_dataoceleba_{0}_set{1}_supervisiono0_traino{2}_ejo10_dupo3'.format(att, fine_num, constrain_nums2[fine_num] if incr == 2 else constrain_nums8[fine_num])
                        except KeyError:
                            continue
                    else:
                        filename = 'pretrain_dataoceleba_{0}_set{1}_supervisiono0_traino{2}_ejo10_dupo3'.format(att, fine_num, train_num)
                else:
                    if '- constrain' in pre_set:
                        try:
                            filename = 'modeloresnet50_weightso{1}_dataoceleba_{0}_set{2}_traino{3}/dupo3'.format(att, pre_set[:pre_set.index(' - constrain')], fine_num, constrain_nums2[fine_num] if incr == 2 else constrain_nums8[fine_num])
                        except KeyError:
                            continue
                    else:
                        filename = 'modeloresnet50_weightso{1}_dataoceleba_{0}_set{2}_traino{3}/dupo3'.format(att, pre_set, fine_num, train_num)

            diffs = []
            f_fprs = []
            m_fprs = []
            aucs = []
            for d in range(dups):
                if train_num == 0 or ('scratch' in pre_set and version not in [3, 4, 7]):
                    file_path = './models/{0}/loss_info.pkl'.format(filename.replace('dupo3', str(d+1)))
                else:
                    file_path = './models/{0}/loss_info.pkl'.format(filename.replace('dupo3', 'dupo{}'.format(d+1)))
                try:
                    loss_info = pickle.load(open(file_path, 'rb'))
                except:
                    print("Missing: {}".format(file_path))
                    continue

                probs = np.array(loss_info['test_probs'][-1])[:, 0]
                labels = np.array(loss_info['test_labels'])[:, 0]

                gens = np.array(loss_info['test_gender'][-1])

                if version in [0, 1]:
                    f = np.where(gens==1)[0]
                    m = np.where(gens==0)[0]
                    has_att = np.where(labels==1)[0]
                    no_att = np.where(labels==0)[0]
                    f_att = np.array(list(set(f)&set(has_att)))
                    m_att = np.array(list(set(m)&set(has_att)))
                    f_noatt = np.array(list(set(f)&set(no_att)))
                    m_noatt = np.array(list(set(m)&set(no_att)))
                    big_prop = (10-incr)*.1
                    small_prop = incr*.1
                    min_len = np.amin([len(m_att)/small_prop, len(f_noatt)/small_prop, len(f_att)/big_prop, len(m_noatt)/big_prop])
                    this_set = np.concatenate([np.random.choice(f_att, int(min_len*(10-incr)*.1), replace=False), np.random.choice(m_noatt, int(min_len*(10-incr)*.1), replace=False), np.random.choice(f_noatt, int(min_len*incr*.1), replace=False), np.random.choice(m_att, int(min_len*incr*.1), replace=False)])
                    labels, probs, gens = labels[this_set], probs[this_set], gens[this_set]
                elif version in [2]: 
                    f = np.where(gens=='Female')[0]
                    m = np.where(gens=='Male')[0]

                    has_att = np.where(labels==1)[0]
                    no_att = np.where(labels==0)[0]
                    f_has = np.array(list(set(f)&set(has_att)))
                    m_has = np.array(list(set(m)&set(has_att)))
                    f_hasnot = np.array(list(set(f)&set(no_att)))
                    m_hasnot = np.array(list(set(m)&set(no_att)))

                    key = '{0}-{1}'.format(att.replace('-', ' '), train_num)
                    train_min_has, train_min_hasnot = saved_props[key]
                    posprop = train_min_has/train_num

                    if att in ['handbag', 'dining-table']:
                        incr = 2
                    elif att in ['chair', 'cup']:
                        incr = 8
                    else:
                        assert NotImplementedError

                    big_prop = (10-incr)*.1
                    small_prop = incr*.1

                    min_has = np.amin([len(f_has)/big_prop, len(m_has)/small_prop])
                    min_hasnot = np.amin([len(f_hasnot)/small_prop, len(m_hasnot)/big_prop])
                    while min_hasnot < int(min_has/posprop) - min_has:
                        min_has -= 1

                    pos_labels = np.concatenate([np.random.choice(f_has, int(min_has*((10-incr)*.1)), replace=False), np.random.choice(m_has, int(min_has*(incr*.1)), replace=False)])
                    neg_labels = np.concatenate([np.random.choice(f_hasnot, int(min_hasnot*(incr*.1)), replace=False), np.random.choice(m_hasnot, int(min_hasnot*((10-incr)*.1)), replace=False)])
                    neg_labels = np.random.choice(neg_labels, int(len(pos_labels)/posprop) - len(pos_labels), replace=False)
                    this_set = np.concatenate([pos_labels, neg_labels])
                    labels, probs, gens = labels[this_set], probs[this_set], gens[this_set]

                auc_score = roc_auc_score(labels, probs)
                aucs.append(auc_score)
                if version in [2]:
                    f = np.where(gens=='Female')[0]
                    m = np.where(gens=='Male')[0]
                elif version in [0, 1]:
                    f = np.where(gens==1)[0]
                    m = np.where(gens==0)[0]
                
                f_auc = roc_auc_score(labels[f], probs[f])
                m_auc = roc_auc_score(labels[m], probs[m])

                threshold = np.sort(probs)[int(-np.sum(labels)-1)]

                val_probs = np.array(loss_info['val_probs'][-1])[:, 0]
                val_labels = np.array(loss_info['val_labels'][-1])[:, 0]
                val_threshold = np.sort(val_probs)[int(-np.sum(val_labels)-1)]
                preds = probs>val_threshold

                for g, gen in enumerate([f, m]):
                    tn, fp, fn, tp = confusion_matrix(labels[gen], preds[gen]).ravel()
                    fpr = fp/(fp+tn)
                    fnr = fn/(fn+tp)
                    tpr = tp/(tp+fn)
                    if g == 0:
                        f_fpr, f_fnr, f_tpr = fpr, fnr, tpr
                    else:
                        m_fpr, m_fnr, m_tpr = fpr, fnr, tpr
                diff = f_tpr - m_tpr
                diffs.append(diff)
                f_fprs.append(f_fpr)
                m_fprs.append(m_fpr)

            key = 'p{0}-f{1}'.format(pre_ind, fine_ind)
            dict_diffs[key] = diffs
            dict_f_fprs[key] = f_fprs
            dict_m_fprs[key] = m_fprs
            dict_aucs[key] = aucs
    best_auc = 0.
    for pre_ind, pre_set in enumerate(pre_sets):
        for fine_ind, fine_set in enumerate(fine_sets):
            key = 'p{0}-f{1}'.format(pre_ind, fine_ind)
            try:
                aucs = dict_aucs[key]
                best_auc = np.amax([best_auc, np.mean(aucs)])
            except KeyError:
                print("Missing {0} on fine {1}".format(pre_set, fine_set))

    for pre_ind, pre_set in enumerate(pre_sets):
        if ('- constrain' not in pre_set) or version == 6:
            axes[a].scatter([], [], c="C{}".format(pre_ind), label=preset_name[pre_set].replace('_', ''))
        xs = []
        ys = []
        xerr = []
        yerr = []
        for fine_ind, fine_set in enumerate(fine_sets):
            key = 'p{0}-f{1}'.format(pre_ind, fine_ind)
            try:
                diffs, f_fprs, m_fprs, aucs = dict_diffs[key], dict_f_fprs[key], dict_m_fprs[key], dict_aucs[key]
            except KeyError:
                continue

            xs.append(np.mean(aucs))
            ys.append(np.mean(diffs))
            xerr.append(1.96*np.std(aucs)/len(aucs))
            yerr.append(1.96*np.std(diffs)/len(diffs))
            min_val = np.amin([min_val, np.mean(aucs)-1.96*np.std(aucs)/len(aucs)])
            max_val = np.amax([max_val, np.mean(aucs)+1.96*np.std(aucs)/len(aucs)])
        s_size = 20
        linestyle = 'solid'
        if '- constrain' in pre_set and version != 6:
            linestyle = 'dotted'
            c = pre_sets.index(pre_set[:pre_set.index(' - constrain')])
        else:
            c = pre_ind
        if version in [2]:
            axes[a].errorbar(xs, ys, xerr=xerr, yerr=yerr, c="C{}".format(c), alpha=.7, zorder=1, fmt='.')
        else:
            axes[a].errorbar(xs, ys, xerr=xerr, yerr=yerr, c="C{}".format(c), alpha=.7, zorder=1, linestyle=linestyle)
        axes[a].scatter(xs, ys, c="C{}".format(c), alpha=.7, zorder=1, s= 15)
        if version in [0, 1]:
            if att in ['Wearing_Earrings', 'Blond_Hair']:
                if version == 0:
                    test_ind = 2
                elif version == 1:
                    test_ind = list(np.sort(list(constrain_nums2.keys()))).index(3)
            elif att in ['Bags_Under_Eyes', 'Brown_Hair']:
                if version == 0:
                    test_ind = 8
                elif version == 1:
                    test_ind = list(np.sort(list(constrain_nums8.keys()))).index(9)
        elif version in [2]:
            if att in ['dining-table', 'bottle']:
                test_ind = 2
            elif att in ['chair', 'cup']:
                test_ind = 8

        if ('- constrain' not in pre_set) or version == 1:
            axes[a].scatter([xs[test_ind]], [ys[test_ind]], marker='o', c="C{}".format(pre_ind), s=s_size, zorder=3)
            axes[a].scatter([xs[test_ind]], [ys[test_ind]], marker='o', c="k", s=(s_size*3), zorder=2)

    axes[a].set_title(att.replace('_', ' ').replace('-', ' ').replace('Bags Under Eyes', 'Eyebags').replace('Wearing Earrings', 'Earrings'))
    axes[a].set_ylabel('FPR Difference')
    axes[a].set_xlabel('AUC')

    if version in [0, 1] and args.two:
        yticks = [-.9, -.6, -.3, 0, .3, .6, .9]
        xticks = [.2, .4, .6, .8, 1.0]
        axes[a].set_yticks(yticks)
        axes[a].set_xticks(xticks)
        axes[a].set_xticklabels(xticks)
        axes[a].set_ylim([-.98, .95])
        axes[a].set_xlim([.2, 1.])

        axes[a].plot([.22, .98], [0, 0], linestyle='dotted', c='k')
        axes[a].legend()
    elif version == 2 and args.two:
        yticks = [-.1, -.05, 0, .05, .1, .15]
        xticks = [.6, .7, .8, .9]
        axes[a].set_yticks(yticks)
        axes[a].set_xticks(xticks)
        axes[a].set_xticklabels(xticks)
        
        if train_num == 1000:
            axes[a].set_xlim([.53, .96])
            axes[a].set_ylim([-.12, .18])
            axes[a].plot([.55, .94], [0, 0], linestyle='dotted', c='k')
        elif train_num == 5000:
            axes[a].set_xlim([.56, .96])
            axes[a].set_ylim([-.12, .17])
            axes[a].plot([.58, .94], [0, 0], linestyle='dotted', c='k')
        #axes[a].legend()
    else:
        axes[a].plot([min_val, max_val], [0, 0], linestyle='dotted', c='k')
        axes[a].legend()
plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('figures/corr_v{0}_n{1}_t{2}.png'.format(version, train_num, 'a' if args.two else 'b'), dpi=200)
plt.close()

