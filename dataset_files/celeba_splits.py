import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os

attributes = ['Wearing_Earrings', 'Brown_Hair', 'Blond_Hair', 'Bags_Under_Eyes']
annotations = pd.read_csv('/Data/CelebA/celeba/list_attr_celeba.txt', delim_whitespace=True, header=1)
filenames = pd.read_csv('/Data/CelebA/celeba/list_eval_partition.txt', delim_whitespace=True, header=None)

imageids = []
genders = []
atts = []
for i in range(len(annotations)):
    if filenames[1].iloc[i] == 0:
        imageid = filenames[0].iloc[i][:-4]
        annots = annotations.iloc[i]
        imageids.append(imageid)
        genders.append(annots['Male']==1)
        these_atts = []
        for attrib in attributes:
            these_atts.append(annots[attrib]==1)
        atts.append(these_atts)
atts = np.array(atts)
genders, imageids = np.array(genders), np.array(imageids)

for a, att in enumerate(attributes):
    f = np.where(genders == 0)[0]
    m = np.where(genders == 1)[0]
    has_att = np.where(atts[:, a]==1)[0]
    no_att = np.where(atts[:, a]==0)[0]
    f_att = np.array(list(set(f)&set(has_att)))
    m_att = np.array(list(set(m)&set(has_att)))
    f_noatt = np.array(list(set(f)&set(no_att)))
    m_noatt = np.array(list(set(m)&set(no_att)))

    half_f = len(f)//2
    half_m = len(m)//2

    min_len = np.amin([len(chunk) for chunk in [f_att, m_att, f_noatt, m_noatt]])

    all_sets = []
    for incr in range(11):
        this_set = np.concatenate([np.random.choice(f_att, int(min_len*(10-incr)*.1), replace=False), np.random.choice(m_noatt, int(min_len*(10-incr)*.1), replace=False), np.random.choice(f_noatt, int(min_len*incr*.1), replace=False), np.random.choice(m_att, int(min_len*incr*.1), replace=False)])
        all_sets.append(this_set)
        these_annotations = atts[this_set, a]

        these_genders = genders[this_set]
        f = np.where(these_genders == 0)[0]
        m = np.where(these_genders == 1)[0]
    pickle.dump(all_sets, open('{0}_incr_splits.pkl'.format(att), 'wb'))

