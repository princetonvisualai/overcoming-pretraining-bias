import pickle
import numpy as np
from pycocotools.coco import COCO
import json

m_words = set(['male', 'boy', 'man', 'gentleman', 'boys', 'men', 'males', "gentlemen"])
f_words = set(['female', 'girl', 'woman', 'lady', 'girls', 'women', 'females', 'ladies'])

for version in ['train', 'val']:
    imageid_to_gender = {}
    annotations = json.load(open('/Data/Coco/2014data/annotations/captions_{}2014.json'.format(version), 'rb'))['annotations']
    imageid_to_caption = {}
    for annot in annotations:
        imageid = annot['image_id']
        if imageid not in imageid_to_caption.keys():
            imageid_to_caption[imageid] = []
        imageid_to_caption[imageid].append(annot['caption'])

    for imageid in imageid_to_caption.keys():
        m_presence = False
        f_presence = False
        for cap in imageid_to_caption[imageid]:
            words = [word.lower().strip() for word in cap.split(' ')]
            if len(set(words)&m_words) > 0:
                m_presence = True
            if len(set(words)&f_words) > 0:
                f_presence = True
        if m_presence and not f_presence:
            imageid_to_gender[imageid] = 'M'
        elif f_presence and not m_presence:
            imageid_to_gender[imageid] = 'F'
        else:
            imageid_to_gender[imageid] = 'None'
    values = list(imageid_to_gender.values())
    pickle.dump(imageid_to_gender, open('coco_{}_gender.pkl'.format(version), 'wb'))

