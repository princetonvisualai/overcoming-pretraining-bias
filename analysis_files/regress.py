import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score, confusion_matrix
import pickle
import sys
sys.path.append('..')
from utils import *

controls = ['Wearing_Hat', 'Chubby', 'Bangs']
attributes = ATTRIBUTES_11 

fem_num, att_to_s = pickle.load(open('att_to_s.pkl', 'rb'))
skewz_val = .8 - fem_num
num = 1024

big_df = []
for a, att in enumerate(attributes):
    for v in [0, 1]:
        if v == 0:
            this_att = att
        elif v == 1:
            this_att = att+'_skewz'
        filenames = ['modeloresnet50_weightsopretrain_dataoceleba_Male_supervisiono0_traino-1_ejo50_dataoceleba_{0}_traino{1}_dupo3'.format(this_att, num if num != 130216 else -1)]
        for control in controls:
            filenames.append('modeloresnet50_weightsopretrain_dataoceleba_{2}_supervisiono0_traino-1_ejo10_dataoceleba_{0}_traino{1}_dupo3'.format(this_att, num if num != 130216 else -1, control))

        saliency, skew = att_to_s[this_att[:-6] if 'skewz' in this_att else this_att]
        vals = [[] for _ in range(len(filenames))]
        if 'skewz' in this_att:
            skew = .8
        row = [a, saliency, np.absolute(skew-fem_num)]
        for n, name in enumerate(filenames):
            aucs = []
            for d in range(5):
                if 'pretrain' in name[:10]:
                    file_path = 'models/{0}/loss_info.pkl'.format(name.replace('dupo3', '{}'.format(d+1)))
                else:
                    file_path = 'models/{0}/loss_info.pkl'.format(name.replace('dupo3', 'dupo{}'.format(d+1)))
                try:
                    loss_info = pickle.load(open(file_path, 'rb'))
                except:
                    print("Missing: {}".format(file_path))
                    continue

                probs = np.array(loss_info['test_probs'][-1])[:, 0]
                labels = np.array(loss_info['test_labels'])[:, 0]
                auc_score = roc_auc_score(labels, probs)
                aucs.append(auc_score)
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
                if 'skewz' in this_att:
                    vals[n].append(f_fpr-m_fpr)
                    labels = np.array(loss_info['train_labels'])
                    labels = labels.squeeze()
                else:
                    if skew_same:
                        if skew > fem_num:
                            vals[n].append(f_fpr-m_fpr)
                        else:
                            vals[n].append(m_fpr-f_fpr)
                    else:
                        vals[n].append(f_fpr-m_fpr)
        base_val = np.amax([np.mean(chunk) for chunk in vals[1:]])
        dot_size = np.mean(vals[0])/base_val
        if np.mean(vals[0]) < 0 and base_val > 0:
            dot_size = 0
        if np.mean(vals[0]) > 0 and base_val < 0:
            dot_size = 5
        row.append(dot_size)
        row.append(np.mean(labels))
        big_df.append(row)

big_df = np.array(big_df)
big_df = pd.DataFrame(big_df, columns=['Attribute', 'Saliency', 'Skew', 'Value', 'Pos_Prop'])
saliency = np.array(big_df['Saliency'])
skew = np.array(big_df['Skew'])
values = np.array(big_df['Value'])
md = smf.mixedlm('Value ~ Saliency + Skew', big_df, groups=big_df['Attribute'])
mdf = md.fit()
print(mdf.summary())
preds = mdf.predict(big_df)
labels = big_df['Value']
score = r2_score(labels, preds)
print("Score: {}".format(score))

