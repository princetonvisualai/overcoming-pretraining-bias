import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix
import sys
sys.path.append('..')
from utils import *

rate = -1
tickfont = 14

these_attributes = BIAS_ATTRIBUTES_4 

fem_num, att_to_s = pickle.load(open('att_to_s.pkl', 'rb'))

controls = ['Wearing_Hat', 'Chubby', 'Bangs']
picked_controls = []
### find the controls for each attribute
num = 1024
for label in these_attributes:
    saliency, skew = att_to_s[label[:-6] if 'skewz' in label else label]
    vals = [[] for _ in range(len(controls))]
    for c, control in enumerate(controls):
        name = 'modeloresnet50_weightsopretrain_dataoceleba_{2}_supervisiono0_traino-1_ejo50_dataoceleba_{0}_traino{1}_dupo3'.format(label, num if num != 130216 else -1, control)
        for d in range(5):
            file_path = 'models/{0}/loss_info.pkl'.format(name.replace('dupo3', 'dupo{}'.format(d+1)))
            try:
                loss_info = pickle.load(open(file_path, 'rb'))
            except:
                print("Missing: {}".format(file_path))
                continue

            probs = np.array(loss_info['test_probs'][-1])[:, 0]
            labels = np.array(loss_info['test_labels'])[:, 0]
            auc_score = roc_auc_score(labels, probs)
            gens = np.array(loss_info['test_gender'][-1])
            f = np.where(gens==1)[0]
            m = np.where(gens==0)[0]

            threshold = np.sort(probs)[int(-np.sum(labels)-1)]
            preds = probs>threshold

            for g, gen in enumerate([f, m]):
                tn, fp, fn, tp = confusion_matrix(labels[gen], preds[gen]).ravel()
                fpr = fp/(fp+tn)
                fnr = fn/(fn+tp)
                if g == 0:
                    f_fpr, f_fnr = fpr, fnr
                else:
                    m_fpr, m_fnr = fpr, fnr
            vals[c].append(auc_score)
    picked_controls.append(controls[np.argmax([np.mean(val) for val in vals])])
print("picked controls: {}".format(picked_controls))

for l_ind, label in enumerate(these_attributes):
    dups = 5
    to_graph = ['AUC', 'FPR Difference']
    save_name = 'att_' + label
    nums = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 130216]
    num_to_means = [[] for _ in range(len(to_graph)*3)]
    f, axes = plt.subplots(len(to_graph), 1, figsize=(3.8, 6)) # length, then height

    for n, num in enumerate(nums): 
        model_names = ['modeloresnet50_weightsopretrain_dataoceleba_Male_supervisiono0_traino-1_ejo50_dataoceleba_{0}_traino{1}/dupo3'.format(label, num if num != 130216 else -1), 'modeloresnet50_weightsopretrain_dataoceleba_Wearing_Hat_supervisiono0_traino-1_ejo50_dataoceleba_{0}_traino{1}/dupo3'.format(label, num if num != 130216 else -1)]
        model_names.append('pretrain_dataoceleba_{0}_supervisiono0_traino{1}_ejo50_dupo3'.format(label, num if num!= 130216 else -1))
        short_names = ['Gendered', 'Control', 'Scratch']

        for n_i, name in enumerate(model_names):
            if 'Wearing_Hat' in name:# rather than max at each time step, take the one that at 1024 has the best one
                name = name.replace('Wearing_Hat', picked_controls[l_ind])

            if n == 0:
                for m in range(len(to_graph)):
                    axes[m].scatter([], [], c='C{}'.format(n_i), label=short_names[n_i])
            values = [[] for _ in range(len(to_graph))] 
            
            for d in range(dups):
                if 'pretrain' in name[:10]:
                    file_path = 'models/{0}/loss_info.pkl'.format(name.replace('dupo3', str(d+1)))
                else:
                    file_path = 'models/{0}/loss_info.pkl'.format(name.replace('dupo3', 'dupo{}'.format(d+1)))

                try:
                    loss_info = pickle.load(open(file_path, 'rb'))
                except:
                    print("File not found: {}".format(file_path))

                probs = np.array(loss_info['test_probs'][-1])
                labels = np.array(loss_info['test_labels'])
                att = np.array(loss_info['test_gender'][-1])
                if rate == -1:
                    rate = np.sum(labels) / len(labels)
                ap = normalized_ap_wrapper(labels, probs, rate=rate, args=None)
                probs = np.array(loss_info['test_probs'][-1])[:, 0]
                labels = np.array(loss_info['test_labels'])[:, 0]
                auc = roc_auc_score(labels, probs)
                values[0].append(auc)
                f = np.where(att==1)[0]
                m = np.where(att==0)[0]

                f_auc = roc_auc_score(labels[f], probs[f])
                m_auc = roc_auc_score(labels[m], probs[m])
                threshold = np.sort(probs)[int(-np.sum(labels)-1)]
                preds = probs>threshold
                for g, gen in enumerate([f, m]):
                    tn, fp, fn, tp = confusion_matrix(labels[gen], preds[gen]).ravel()
                    fpr = fp/(fp+tn)
                    fnr = fn/(fn+tp)
                    if g == 0:
                        f_fpr, f_fnr = fpr, fnr
                    else:
                        m_fpr, m_fnr = fpr, fnr

                values[1].append(f_fpr-m_fpr)
            for m in range(len(to_graph)):
                axes[m].errorbar(num, np.mean(values[m]), yerr=1.96*np.std(values[m])/len(values[m]), color='C{}'.format(n_i), alpha=.6)
                num_to_means[len(model_names)*m+n_i].append(np.mean(values[m]))

    for m in range(len(to_graph)):
        axes[m].set_xscale('log', base=2)
        axes[m].set_ylabel(to_graph[m], fontsize=tickfont)
        axes[m].tick_params(axis='y', labelsize=tickfont)
        axes[m].tick_params(axis='x', labelsize=tickfont)
        #if m == 0:
        #    axes[m].legend()
        for n_i in range(len(model_names)):
            axes[m].plot(nums, num_to_means[len(model_names)*m+n_i], c='C{}'.format(n_i), alpha=.6) 

    plt.tight_layout()
    plt.savefig('figures/xaxis_trainnum_{}.png'.format(save_name), dpi=200)
    plt.close()




