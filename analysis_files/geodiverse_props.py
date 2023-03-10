import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import copy
import sys
sys.path.append('..')
from utils import *

model_names = ['pytorch_imagenet1k_v2', 'moco', 'simclr', 'places', 'scratch']
name_mapping = {'pretrain_geode': 'Geode', 'pretrain_imagenet': 'ImageNet'}

model_names = ['pretrain_geode', 'pretrain_imagenet']
train_nums = [128]
dataset_name = ['coco_dsset', '']
fine_sets = np.arange(11)

f, axes = plt.subplots(len(train_nums), 2, figsize=(5.7, 1.8)) # length, then height

income_cats = [1, 2, 3]

min_val = [1, 1]
max_val = [0, 0]

for t, train_num in enumerate(train_nums):
    for m, model in enumerate(model_names):
        yvals = []
        yerrs = []
        for fine_ind, fine_set in enumerate(fine_sets):
            values_for_income = [[] for _ in range(3)]
            ds_perf = []
            coco_perf = []
            for d in range(1, 6):
                if 'pretrain_' in model:
                    file_path = 'models/modeloresnet50_weightsomodeloresnet50_weightsoscratch_datao{5}_traino-17dupo0_datao{0}{2}{1}_traino{3}/dupo{4}'.format(dataset_name[0], dataset_name[1], fine_set, train_num, d, model[model.index('pretrain_')+len('pretrain_'):])
                else:
                    file_path = 'models/modeloresnet50_weightso{5}_datao{0}{2}{1}_traino{3}/dupo{4}'.format(dataset_name[0], dataset_name[1], fine_set, train_num, d, model)

                try:
                    loss_info = pickle.load(open(file_path+'/loss_info.pkl', 'rb'))
                except:
                    if model == 'scratch':
                        try:
                            file_path = 'models/modeloresnet50_weightso{5}_datao{0}{2}{1}_traino{3}/dupo{4}'.format(dataset_name[0], dataset_name[1], fine_set, train_num, d, model)
                            loss_info = pickle.load(open(file_path+'/loss_info.pkl', 'rb'))
                        except:
                            print("Missing: {}".format(file_path))
                            continue
                    else:
                        print("Missing: {}".format(file_path))
                        continue

                coco_labels = np.array(loss_info['test_labels'])
                coco_probs = np.array(loss_info['test_probs'])[-1]
                ds_country = np.array(loss_info['ds_country'][-1])
                ds_region = np.array(loss_info['ds_region'])[-1]
                ds_income = np.array(loss_info['ds_income'])[-1]
                ds_labels = np.array(loss_info['ds_labels'])
                ds_probs = np.array(loss_info['ds_probs'])
                ds_probs = ds_probs[-1]

                changed_income = [round(np.log(float(income))/3) for income in ds_income]
                region_cats = list(np.sort(np.unique(ds_region)))

                N = np.sum(coco_labels)/len(coco_labels[0])
                all_aps = [normalized_ap(coco_labels[:, i], coco_probs[:, i], N=N) for i in range(len(coco_labels[0]))]
                ap_score = average_precision_score(coco_labels, coco_probs)
                coco_perf.append(ap_score)

                this_income_to_hr = [[] for _ in range(len(income_cats))]
                for ind in np.arange(len(ds_probs)):
                    top5 = np.argsort(ds_probs[ind])[-5:]
                    correct = np.where(ds_labels[ind])[0]
                    income = income_cats.index(changed_income[ind])
                    region = region_cats.index(ds_region[ind])
                    if correct in top5:
                        this_income_to_hr[income].append(1)
                    else:
                        this_income_to_hr[income].append(0)

                this_ds_perf = [0, 0]
                for i in range(len(values_for_income)):
                    values_for_income[i].append(np.mean(this_income_to_hr[i]))
                    this_ds_perf[0] += np.sum(this_income_to_hr[i])
                    this_ds_perf[1] += len(this_income_to_hr[i])
                ds_perf.append(this_ds_perf[0]/this_ds_perf[1])
            yvals.append([np.mean(chunk) for chunk in [ds_perf, coco_perf]])
            yerrs.append([1.96*np.std(chunk)/len(chunk) for chunk in [ds_perf, coco_perf]])

        yvals, yerrs = np.array(yvals), np.array(yerrs)
        for i in [0, 1]:
            min_val[i] = np.amin([min_val[i], np.nanmin(yvals[:, i]-yerrs[:, i])])
            max_val[i] = np.amax([max_val[i], np.nanmax(yvals[:, i]+yerrs[:, i])])
        if m == 0:
            axes[0].set_title('Dollar Street', fontsize=9)
            axes[1].set_title('COCO', fontsize=9)
        s_size = 20
        if 'geode' in model:
            axes[0].plot(fine_sets, [yvals[9, 0]]*len(fine_sets), c='C{}'.format(m),  linestyle='dashed', zorder=1) 
            axes[1].errorbar(fine_sets, yvals[:, 1], yerr=yerrs[:, 1], c='C{}'.format(m), zorder=1) 
            axes[0].scatter([fine_sets[1]], [yvals[9, 0]], marker='o', c="C{}".format(m), s=s_size, zorder=3)
        elif 'imagenet' in model:
            axes[1].plot(fine_sets, [yvals[1, 1]]*len(fine_sets), c='C{}'.format(m),  linestyle='dashed', zorder=1) 
            axes[0].errorbar(fine_sets[::-1], yvals[:, 0], yerr=yerrs[:, 0], c='C{}'.format(m), zorder=1) 
            axes[1].scatter([fine_sets[1]], [yvals[1, 1]], marker='o', c="C{}".format(m), s=s_size, zorder=3)

tickfont = 9
titlefont = 9
for i in [0, 1]:
    axes[i].set_xticks([0, 2, 4, 6, 8, 10])
    axes[i].set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=tickfont)
    if i == 0:
        yticks = [.1, .2, .3, .4, .5]
        yticklabels = yticks
    elif i == 1:
        yticks = [.044, .045, .046, .047, .048, .049]
        yticklabels = [round(chunk, 1) for chunk in 100*np.array(yticks)]
    axes[i].set_yticks(yticks)
    axes[i].set_yticklabels(yticklabels, fontsize=tickfont)
    if version == 8 and i == 1:
        axes[i].set_xlabel('Percentage of Data from COCO', fontsize=titlefont)
    else:
        axes[i].set_xlabel('Percentage of Data from Dollar Street', fontsize=titlefont)

axes[0].set_ylabel('Accuracy', fontsize=titlefont)
axes[1].set_ylabel('Norm AP (x100%)', fontsize=titlefont)

axes[1].plot([], [], c='C0', label='GeoDE', linestyle='dashed')
axes[1].plot([], [], c='C1', label='ImageNet')
axes[1].legend(prop={"size":tickfont})

plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('./figures/geodiverse_props.png'.format(version), dpi=300)
plt.close()


