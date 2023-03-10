import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, auc, roc_auc_score
import copy
import statsmodels.formula.api as smf
import pandas as pd
from patsy.contrasts import Treatment
import argparse
import sys
sys.path.append('..')
from utils import *

saliences = pickle.load(open('geo_sals.pkl', 'rb'))[0]

pair_to_setnum = {}
pair_to_vals = {}

dups = 5
set_dists = np.arange(1, 10) 
trainnums = [128, 1024]
props = [5, 5]
all_cats = []

for pair in (GREATER_PAIRS+LESS_PAIRS+SAME_PAIRS):
    lab1, lab2 = pair
    cat = '{0}-{1}'.format(lab1, lab2)
    if cat not in saliences.keys():
        cat = '{0}-{1}'.format(lab2, lab1)
    all_cats.append(cat)
print(all_cats)

for t_ind, trainnum in enumerate(trainnums):
    for cat in all_cats:
        lab1, lab2 = cat.split('-')
        if cat not in saliences.keys():
            assert 'Hair' in cat
            sal = 0
        else:
            sal = saliences[cat]
        sal += .5

        for p_ind, prop in enumerate(props):
            all_values1 = []
            all_values2 = []
            all_values3 = []
            all_values4 = []

            for set_ind, set_dist in enumerate(set_dists):
                values1 = []
                values2 = []
                for d in range(dups):
                    file_path = 'models/modeloresnet50_weightsopretrain_dataofairface_{0}_supervisiono0_traino-1_ejo50_dataoceleba_{0}+{1}+set{3}-prop{5}_traino{4}/dupo{2}/loss_info.pkl'.format(lab1, lab2, d+1, set_dist, trainnum, prop)
                    try:
                        loss_info = pickle.load(open(file_path, 'rb'))
                    except:
                        print("Missing: {}".format(file_path))
                        continue

                    probs = np.array(loss_info['test_probs'][-1])[:, 0]
                    labels = np.array(loss_info['test_labels'])[:, 0]
                    att1 = np.array(loss_info['test_young'][-1])
                    att2 = np.array(loss_info['test_gender'][-1])

                    both_neg = set(np.where(labels==0)[0])
                    keep_att1 = set(np.where(att1==1)[0])&set(np.where(att2==0)[0])
                    keep_att2 = set(np.where(att2==1)[0])&set(np.where(att1==0)[0])

                    keep_att1 = np.array(list(both_neg.union(keep_att1)))
                    keep_att2 = np.array(list(both_neg.union(keep_att2)))

                    overall = roc_auc_score(labels, probs)
                    score1 = roc_auc_score(att1[keep_att1], probs[keep_att1])
                    score2 = roc_auc_score(att2[keep_att2], probs[keep_att2])
                    values1.append(score1)
                    values2.append(score2)
                
                values3 = []
                values4 = []
                for d in range(dups):
                    file_path = 'models/modeloresnet50_weightsopretrain_dataofairface_{0}_supervisiono0_traino-1_ejo50_dataoceleba_{0}+{1}+set{3}-prop{5}_traino{4}/dupo{2}/loss_info.pkl'.format(lab2, lab1, d+1, 10-set_dist, trainnum, prop)
                    try:
                        loss_info = pickle.load(open(file_path, 'rb'))
                    except:
                        print("Missing: {}".format(file_path))
                        continue

                    att2 = np.array(loss_info['test_young'][-1])
                    att1 = np.array(loss_info['test_gender'][-1])

                    probs = np.array(loss_info['test_probs'][-1])[:, 0]
                    labels = np.array(loss_info['test_labels'])[:, 0]
                    overall = roc_auc_score(labels, probs)


                    both_neg = set(np.where(labels==0)[0])
                    keep_att1 = set(np.where(att1==1)[0])&set(np.where(att2==0)[0])
                    keep_att2 = set(np.where(att2==1)[0])&set(np.where(att1==0)[0])

                    keep_att1 = np.array(list(both_neg.union(keep_att1)))
                    keep_att2 = np.array(list(both_neg.union(keep_att2)))
                    score1 = roc_auc_score(att1[keep_att1], probs[keep_att1])
                    score2 = roc_auc_score(att2[keep_att2], probs[keep_att2])
                    values3.append(score1)
                    values4.append(score2)

                all_values1.append(values1)
                all_values2.append(values2)
                all_values3.append(values3)
                all_values4.append(values4)

            try:
                all_values1, all_values2, all_values3, all_values4 = np.array(all_values1), np.array(all_values2), np.array(all_values3), np.array(all_values4)
            except:
                pass

            def get_spot(pretrained_on, pretrained_off, set_num, rev=False):
                if rev:
                    pretrain = [np.mean(chunk) for chunk in pretrained_on][list(np.array(set_dists)[::-1]).index(set_num)]
                else:
                    pretrain = [np.mean(chunk) for chunk in pretrained_on][set_dists.index(set_num)]
                finetune = np.array([np.mean(chunk) for chunk in pretrained_off])[::-1] > pretrain
                above = np.where(finetune==1)[0]
                if len(above) > 0:
                    spot = set_dists[np.amax(above)]
                else:
                    spot = -1
                return spot
            pair_to_setnum['{0}-{1}'.format(lab1, lab2)] = [sal, get_spot(all_values1, all_values3, 9), get_spot(all_values1, all_values3, 7)]

            pair_to_setnum['{1}-{0}'.format(lab1, lab2)] = [-sal+1., get_spot(all_values4, all_values2, 9, rev=True), get_spot(all_values4, all_values2, 7, rev=True)]
            pair_to_vals['{0}-{1}&{2}'.format(lab1, lab2, trainnum)] = [sal, all_values1, all_values3, False]
            pair_to_vals['{1}-{0}&{2}'.format(lab1, lab2, trainnum)] = [-sal+1., all_values4, all_values2, True]


min_val, max_val = 1, 0
f, axes = plt.subplots(3, len(trainnums), figsize=(8, 10)) # length, then height
diffs_at_5_per_num = [[] for _ in range(len(trainnums))]
diffs_at_1_per_numsal = {}
diffs_at_5_per_numsal = {}
diffs_at_9_per_numsal = {}
for num in trainnums:
    for s in range(3):
        diffs_at_1_per_numsal['{0}-{1}'.format(num, s)] = []
        diffs_at_5_per_numsal['{0}-{1}'.format(num, s)] = []
        diffs_at_9_per_numsal['{0}-{1}'.format(num, s)] = []

set_num = 9
h_counter = 0
hair_num = {}
all_spots = [[] for _ in range(len(trainnums))]
for k, key in enumerate(pair_to_vals.keys()):
    sal, pretrained_on, pretrained_off, rev = pair_to_vals[key]
    key, num = key.split('&')
    num = int(num)
    if key in same:
        ax = 1
        c = same.index(key)
    elif key in greater:
        ax = 2
        c = greater.index(key)
    elif key in less:
        ax = 0
        c = less.index(key)
    else:
        continue

    ys = np.array([np.mean(chunk) for chunk in pretrained_off])
    ys_on = np.array([np.mean(chunk) for chunk in pretrained_on])
    diff = ys_on - ys
    yerrs = np.array([1.96*np.std(chunk)/np.sqrt(len(chunk)) for chunk in pretrained_off])
    if rev:
        pretrain_val = ys_on[list(np.array(set_dists)[::-1]).index(set_num)]
        axes[ax][trainnums.index(num)].errorbar(10-np.array(set_dists), ys, yerr=yerrs, c='C{}'.format(c), label=key, zorder=1)

        finetune = ys > pretrain_val
        above = np.where(finetune==1)[0]
        if len(above) > 0:
            spot = 10 - set_dists[np.amin(above)]
        else:
            spot = -1
        diff5 = diff[list(10-np.array(set_dists)).index(5)]
        diff1 = diff[list(10-np.array(set_dists)).index(1)]
        diff9 = diff[list(10-np.array(set_dists)).index(9)]

    else:
        pretrain_val = ys_on[set_dists.index(set_num)]
        axes[ax][trainnums.index(num)].errorbar(np.array(set_dists), ys, yerr=yerrs, c='C{}'.format(c), label=key, zorder=1)

        finetune = ys > pretrain_val
        above = np.where(finetune==1)[0]
        if len(above) > 0:
            spot = set_dists[np.amax(above)]
        else:
            spot = -1
        diff5 = diff[set_dists.index(5)]
        diff1 = diff[set_dists.index(1)]
        diff9 = diff[set_dists.index(9)]
    axes[ax][trainnums.index(num)].plot(set_dists, [pretrain_val]*len(set_dists), c='C{}'.format(c), linestyle='dotted', zorder=1)
    s_size = 15
    if spot != -1:
        if rev:
            axes[ax][trainnums.index(num)].scatter([spot], [ys[set_dists.index(spot)]], c='C{}'.format(c), s=s_size, zorder=3)
            axes[ax][trainnums.index(num)].scatter([spot], [ys[set_dists.index(spot)]], c='k', s=s_size*3, zorder=2)
        else:
            axes[ax][trainnums.index(num)].scatter([spot], [ys[set_dists.index(spot)]], c='C{}'.format(c), s=s_size, zorder=3)
            axes[ax][trainnums.index(num)].scatter([spot], [ys[set_dists.index(spot)]], c='k', s=s_size*3, zorder=2)
    if key == focus:
        focus_vals = [ys, yerrs, spot, rev, pretrain_val]
    min_val = np.amin([min_val, np.amin(ys-yerrs)])
    max_val = np.amax([max_val, np.amax(ys+yerrs)])
    diffs_at_5_per_num[trainnums.index(num)].append(diff5)
    diffs_at_1_per_numsal['{0}-{1}'.format(num, ax)].append(diff1)
    diffs_at_5_per_numsal['{0}-{1}'.format(num, ax)].append(diff5)
    diffs_at_9_per_numsal['{0}-{1}'.format(num, ax)].append(diff9)

    all_spots[trainnums.index(num)].append(spot)


for n, num in enumerate(trainnums):
    print("Num={0} has {1:.4f} +- {2:.4f}".format(num, np.mean(diffs_at_5_per_num[n]), 1.96*np.std(diffs_at_5_per_num[n])/len(diffs_at_5_per_num[n])))

print("\nSkew = 1")
for num in trainnums:
    for s in range(3):
        print("Num={0}, sal={1}  has {2:.4f} +- {3:.4f}".format(num, s, np.mean(diffs_at_1_per_numsal['{0}-{1}'.format(num, s)]), 1.96*np.std(diffs_at_1_per_numsal['{0}-{1}'.format(num, s)])/len(diffs_at_1_per_numsal['{0}-{1}'.format(num, s)])))
print("\nSkew = 5")
for num in trainnums:
    for s in range(3):
        print("Num={0}, sal={1}  has {2:.4f} +- {3:.4f}".format(num, s, np.mean(diffs_at_5_per_numsal['{0}-{1}'.format(num, s)]), 1.96*np.std(diffs_at_5_per_numsal['{0}-{1}'.format(num, s)])/len(diffs_at_5_per_numsal['{0}-{1}'.format(num, s)])))
print("\nSkew = 9")
for num in trainnums:
    for s in range(3):
        print("Num={0}, sal={1}  has {2:.4f} +- {3:.4f}".format(num, s, np.mean(diffs_at_9_per_numsal['{0}-{1}'.format(num, s)]), 1.96*np.std(diffs_at_9_per_numsal['{0}-{1}'.format(num, s)])/len(diffs_at_9_per_numsal['{0}-{1}'.format(num, s)])))


for a in range(len(axes)):
    axes[a][0].legend()
axes[0][0].set_ylabel('A less salient than B')
axes[1][0].set_ylabel('A equally salient to B')
axes[2][0].set_ylabel('A more salient than B')
for t in range(len(trainnums)):
    axes[0][t].set_title('Train num: {}'.format(trainnums[t]))

for t in range(len(trainnums)):
    for a in range(len(axes)):
        axes[a][t].set_xticks(set_dists)
        axes[a][t].set_xticklabels(['{}%'.format(round(setdist*10)) for setdist in set_dists])
        axes[a][t].set_ylim([.3, 1.1])

plt.tight_layout()
plt.savefig('figures/attribute_pairs.png', dpi=200)
plt.close()

