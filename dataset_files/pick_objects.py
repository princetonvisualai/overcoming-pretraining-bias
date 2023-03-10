import pickle
import numpy as np
from pycocotools.coco import COCO

labels_to_names, categories = pickle.load(open('coco_labels.pkl', 'rb'))
categories = np.array(categories)
names_to_labels = {v: k for k, v in labels_to_names.items()}

version = 'train'
derived_gender = pickle.load(open('dataset_files/coco_{}_gender.pkl'.format(version), 'rb'))
coco = COCO('/Data/Coco/2014data/annotations/instances_{}2014.json'.format(version))
ids = list(coco.anns.keys())
image_ids = np.array(list(set([coco.anns[this_id]['image_id'] for this_id in ids])))
cats = coco.loadCats(coco.getCatIds())
labels_to_names = {}
for cat in cats:
    labels_to_names[cat['id']] = cat['name']

categories = list(labels_to_names.keys())
features = []
gendered_imageids = []
genders = []

for imageid in image_ids:
    if derived_gender[imageid] == 'None':
        continue

    target = np.zeros(len(cats))
    annIds = coco.getAnnIds(imgIds=imageid);
    coco_anns = coco.loadAnns(annIds)
    for ann in coco_anns:
        target[categories.index(ann['category_id'])] = 1

    gend = derived_gender[imageid] == 'F'
    genders.append(gend)
    gendered_imageids.append(imageid)
    features.append(target)

features = np.array(features)
genders, gendered_imageids = np.array(genders), np.array(gendered_imageids)
pickle.dump(gendered_imageids, open('gendered_imageids.pkl', 'wb'))
f = np.where(genders==1)[0]
m = np.where(genders==0)[0]

print("Most represented objects")
for obj in np.argsort(np.sum(features, axis=0))[::-1][:40]:
    print("{0}: {1} ({2:.2f}%)".format(labels_to_names[categories[obj]], np.sum(features, axis=0)[obj], 100*np.mean(features, axis=0)[obj]))

saved_indices = {}
saved_props = {}

for trainnum in [1000, 5000]:
    full_num = int(trainnum / .8)
    for obj in ['handbag', 'chair', 'dining table', 'cup']:
        obj_ind = list(categories).index(names_to_labels[obj])
        m_has = np.where(features[m, obj_ind])[0]
        f_has = np.where(features[f, obj_ind])[0]
        min_has = np.amin([np.amin([len(m_has), len(f_has)]), full_num//2])
        hasnot = np.where(features[:, obj_ind]==0)[0]
        m_hasnot = np.array(list(set(m)&set(hasnot)))
        f_hasnot = np.array(list(set(f)&set(hasnot)))
        min_hasnot = np.amin([len(m_hasnot), len(f_hasnot)])
        key = '{0}-{1}'.format(obj, trainnum)
        saved_indices[key] = []
        saved_props[key] = [min_has, min_hasnot]

        for incr in range(11): # 0 is all f, 10 is all m, then in between is gradient
            pos_labels = np.concatenate([np.random.choice(f_has, int(min_has*((10-incr)*.1)), replace=False), np.random.choice(m_has, int(min_has*(incr*.1)), replace=False)])
            neg_labels = np.concatenate([np.random.choice(f_hasnot, int(min_hasnot*(incr*.1)), replace=False), np.random.choice(m_hasnot, int(min_hasnot*((10-incr)*.1)), replace=False)])
            neg_labels = np.random.choice(neg_labels, full_num-len(pos_labels), replace=False)
            all_labels = np.concatenate([pos_labels, neg_labels])
            saved_indices[key].append(all_labels)
pickle.dump(saved_indices, open('obj_spec_indices.pkl', 'wb'))
pickle.dump(saved_props, open('obj_spec_props.pkl', 'wb'))


