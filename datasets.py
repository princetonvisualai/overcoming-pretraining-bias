import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image
import csv
import numpy as np
import os
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import torch
import xml.etree.ElementTree as ET
import re
from lxml import etree
import cv2
import json
import os
from torchvision import transforms
import xml.etree.ElementTree as ET
from lxml import etree
from scipy.io import loadmat

class CoCoDataset(data.Dataset):

    def __init__(self, transform, version='train', cat6=False, pred_gen=False, variation='', obj=None, setnum=None, train_num=None, ds=False): 
        self.transform = transform
        self.version = version
        self.pred_gen = pred_gen
        self.variation = variation

        self.obj = obj
        self.setnum = setnum
        self.ds = ds
        self.trainnum = train_num

        self.img_folder = '/Data/Coco/2014data/{}2014'.format(version)
        self.coco = COCO('/Data/Coco/2014data/annotations/instances_{}2014.json'.format(version))

        if self.version == 'val':
            dem_data = pd.read_csv('/Data/images_val2014.csv')
            self.gender_info = {int(dem_data.iloc[i]['id']): str(dem_data.iloc[i]['bb_gender']) for i in range(len(dem_data))}
            self.skin_info = {int(dem_data.iloc[i]['id']): str(dem_data.iloc[i]['bb_skin']) for i in range(len(dem_data))}
        self.derived_gender = pickle.load(open('dataset_files/coco_{}_gender.pkl'.format(version), 'rb'))


        ids = list(self.coco.anns.keys())
        self.image_ids = np.array(list(set([self.coco.anns[this_id]['image_id'] for this_id in ids])))
        if self.pred_gen:
            nones = 0
            new_imageids = []
            for imageid in self.image_ids:
                if self.derived_gender[imageid] == 'None':
                    nones += 1
                else:
                    new_imageids.append(imageid)
            self.image_ids = np.array(new_imageids)
            assert nones > 100

        if 'salient' in self.variation:
            if self.version == 'val':
                self.image_ids = pickle.load(open('dataset_files/justgenders_val.pkl', 'rb'))
            else:
                self.image_ids = pickle.load(open('dataset_files/justgenders.pkl', 'rb'))

        if 'skew' in self.variation:
            if self.version != 'val':
                self.image_ids = pickle.load(open('dataset_files/skew_sets.pkl', 'rb'))[int(self.variation[self.variation.index('skew')+4:])-1]

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.labels_to_names = {}
        for cat in cats:
            self.labels_to_names[cat['id']] = cat['name']

        self.categories = list(self.labels_to_names.keys())
        labels_to_names, categories = pickle.load(open('dataset_files/coco_labels.pkl', 'rb'))
        assert categories == self.categories
        names_to_labels = {v: k for k, v in labels_to_names.items()}

        if self.setnum is not None:
            assert (self.obj is not None) or (self.ds)

        if self.ds:
            assert self.setnum is not None
            assert train_num is not None
            self.ds_img_folder = '/Data/DollarStreet/MoreDollarStreet/images'

            meta_data = json.load(open('/Data/DollarStreet/MoreDollarStreet/metadata_full_dollar_street.json'))
            ds_to_coco = {'alcoholic_drinks': ['bottle', 'wine glass'], 'armchairs': ['chair', 'couch'], 'beds': ['bed'], 'kids_bed': ['bed'], 'books': ['book'], 'bowls': ['bowl'], 'cars': ['car'], 'computers': ['laptop'], 'cups': ['cup'], 'cutlery': ['fork', 'knife', 'spoon'], 'freezers': ['refrigerator'], 'fruits': ['banana', 'apple', 'orange'], 'kitchen_sinks': ['sink'], 'motorcycles': ['motorcycle'], 'ovens': ['oven'], 'phones': ['cell phone'], 'refrigerators': ['refrigerator'], 'sofas': ['couch'], 'tvs': ['tv'], 'toilets': ['toilet'], 'toothbrushes': ['toothbrush'], 'wall_clocks': ['clock']}
            more_clear = ['beds', 'kids_bed', 'books', 'bowls', 'cars', 'cups', 'kitchen_sinks', 'motorcycles', 'ovens', 'refrigerators', 'sofas', 'tvs', 'toilets', 'toothbrushes', 'wall_clocks']

            mapping, categories = pickle.load(open('dataset_files/coco_labels.pkl', 'rb'))
            named_categories = [mapping[cat] for cat in categories]

            self.ds_image_ids = []
            self.ds_imageid_to_label = {}
            setind = int(self.setnum.split('+')[0])
            onlyincome1 = False
            if len(self.setnum.split('+')) > 1:
                assert self.setnum.split('+')[1] == 'income1'
                onlyincome1 = True

            for meta in meta_data:
                country = meta['country']
                region = meta['region']
                label = meta['label']
                income = meta['income']
                the_id = meta['id']
                if label in more_clear:
                    self.ds_image_ids.append(the_id)
                    cats = np.zeros(80)
                    cats[[named_categories.index(lab) for lab in ds_to_coco[label]]] = 1
                    self.ds_imageid_to_label[the_id] = [cats, country, region, income]
            
            incomes = np.array([round(np.log(float(self.ds_imageid_to_label[imageid][3]))/3) for imageid in self.ds_image_ids]) ## are all 1-3
            incomes1 = np.where(incomes==1)[0]
            ds_indices = np.array(self.ds_image_ids)
            min_len = train_num
            if onlyincome1:
                ds_indices = np.array(self.ds_image_ids)[incomes1]
            self.image_ids = np.concatenate([['ds'+str(ind) for ind in np.random.choice(ds_indices, size=int((10-setind)*.1*min_len), replace=False)], ['co'+str(ind) for ind in np.random.choice(self.image_ids, size=int(setind*.1*min_len), replace=False)]])

        if self.obj is not None and self.setnum is not None:
            incr_sets = pickle.load(open('dataset_files/obj_spec_indices.pkl', 'rb'))['{0}-{1}'.format(self.obj.replace('-', ' '), self.trainnum)]
            self.image_ids = pickle.load(open('dataset_files/gendered_imageids.pkl', 'rb'))[incr_sets[int(self.setnum)]]

        if self.obj is not None:
            self.obj = list(categories).index(names_to_labels[self.obj.replace('-', ' ')])

        self.cat6 = cat6

        if cat6:
            openimage_mapping = pd.read_csv('/Data/OpenImages/class-descriptions-boxable.csv', header=0, names=['code', 'label'])
            categories = list(openimage_mapping['code'])
            split_a, split_b = pickle.load(open('openimage_dataset/corr_split_labels.pkl', 'rb'))
            cat_indices_a = np.array([categories.index(sa) for sa in split_a])
            cat_indices_b = np.array([categories.index(sb) for sb in split_b])
            label_a, label_b = openimage_mapping['label'].iloc[cat_indices_a], openimage_mapping['label'].iloc[cat_indices_b]
            mapping, categories = pickle.load(open('dataset_files/coco_labels.pkl', 'rb'))
            reverse_mapping = {v: k for k, v in mapping.items()}
            cat_indices_a = np.array([categories.index(reverse_mapping[sa.lower().strip()]) for sa in label_a])
            cat_indices_b = np.array([categories.index(reverse_mapping[sb.lower().strip()]) for sb in label_b])
            self.indices = np.concatenate([cat_indices_a, cat_indices_b])

        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.ds:
            if 'co' == image_id[:2]:
                path = self.coco.loadImgs(int(image_id[2:]))[0]["file_name"]

                file_path = os.path.join(self.img_folder, path)
                return self.from_path(file_path)
            elif 'ds' == image_id[:2]:
                file_path = os.path.join(self.ds_img_folder, image_id[2:]) + '.jpg'
                image = Image.open(file_path).convert("RGB")
                image = self.transform(image)

                anns = self.ds_imageid_to_label[image_id[2:]]
                return image, anns[0]
            else:
                assert False

        path = self.coco.loadImgs(int(image_id))[0]["file_name"]

        file_path = os.path.join(self.img_folder, path)
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)


    def from_path(self, file_path):
        image_id = int(os.path.basename(file_path)[-16:-4])

        image = np.array(Image.open(file_path).convert("RGB"))
        target = np.zeros(len(self.categories))

        annIds = self.coco.getAnnIds(imgIds=image_id);
        coco_anns = self.coco.loadAnns(annIds) # coco is [x, y, width, height]
        for ann in coco_anns:
            target[self.categories.index(ann['category_id'])] = 1

        if self.cat6:
            target = target[self.indices]

        if self.transform is None:
            image = image_id
        else:
            if 'salient' in self.variation:
                original_image = copy.deepcopy(image)
                image = Image.fromarray(image)
                if 'salient1' in self.variation or 'salient2' in self.variation:
                    blur = transforms.GaussianBlur(9, sigma=5)
                elif 'salient3' in self.variation or 'salient4' in self.variation:
                    blur = transforms.GaussianBlur(13, sigma=15)
                elif 'salient5' in self.variation or 'salient6' in self.variation:
                    blur = transforms.GaussianBlur(15, sigma=20)
                else:
                    assert NotImplementedError
                image = np.array(blur(image))

                all_people_masks = np.ones((image.shape[0], image.shape[1])) # will be 0's where all the people are
                for ann in coco_anns:
                    if ann['category_id'] == 1:
                        all_people_masks = all_people_masks * (1. - self.coco.annToMask(ann))
                if 'salient1' in self.variation or 'salient3' in self.variation or 'salient5' in self.variation:
                    image = (image*np.expand_dims(all_people_masks, 2)) + (original_image*np.expand_dims(1-all_people_masks, 2))
                elif 'salient2' in self.variation or 'salient4' in self.variation or 'salient6' in self.variation:
                    image = (image*np.expand_dims(1-all_people_masks, 2)) + (original_image*np.expand_dims(all_people_masks, 2))
                else:
                    assert NotImplementedError
                image = image.astype(np.uint8)

            image = Image.fromarray(image)
            image = self.transform(image)

        if self.pred_gen:
            target = np.array([1]) if self.derived_gender[image_id] == 'F' else np.array([0])

        if self.obj:
            target = np.array([target[self.obj]])

        if self.version == 'val':
            try:
                return image, target, [self.gender_info[image_id], self.skin_info[image_id]]
            except KeyError:
                return image, target, ["", ""]
        return image, target

class DollarStreetDataset(data.Dataset):
    
    def __init__(self, transform, version = '', setnum=None): 
        self.version = version 
        self.setnum = setnum

        self.transform = transform
        self.img_folder = '/Data/DollarStreet/MoreDollarStreet/images'
        self.setup_anns()


    def __getitem__(self, index):
        image_id = self.image_ids[index]

        file_path = os.path.join(self.img_folder, image_id) + '.jpg'

        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[:-4]
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)

        anns = self.imageid_to_label[image_id]
        if self.version == 'test':
            return image, anns[0], anns[1:]
        else:
            return image, anns[0]

    def setup_anns(self):
        meta_data = json.load(open('/Data/DollarStreet/MoreDollarStreet/metadata_full_dollar_street.json'))
        ds_to_coco = {'alcoholic_drinks': ['bottle', 'wine glass'], 'armchairs': ['chair', 'couch'], 'beds': ['bed'], 'kids_bed': ['bed'], 'books': ['book'], 'bowls': ['bowl'], 'cars': ['car'], 'computers': ['laptop'], 'cups': ['cup'], 'cutlery': ['fork', 'knife', 'spoon'], 'freezers': ['refrigerator'], 'fruits': ['banana', 'apple', 'orange'], 'kitchen_sinks': ['sink'], 'motorcycles': ['motorcycle'], 'ovens': ['oven'], 'phones': ['cell phone'], 'refrigerators': ['refrigerator'], 'sofas': ['couch'], 'tvs': ['tv'], 'toilets': ['toilet'], 'toothbrushes': ['toothbrush'], 'wall_clocks': ['clock']}
        more_clear = ['beds', 'kids_bed', 'books', 'bowls', 'cars', 'cups', 'kitchen_sinks', 'motorcycles', 'ovens', 'refrigerators', 'sofas', 'tvs', 'toilets', 'toothbrushes', 'wall_clocks']

        mapping, categories = pickle.load(open('dataset_files/coco_labels.pkl', 'rb'))
        named_categories = [mapping[cat] for cat in categories]

        self.image_ids = []
        self.imageid_to_label = {}

        for meta in meta_data:
            country = meta['country']
            region = meta['region']
            label = meta['label']
            income = meta['income']
            the_id = meta['id']
            if label in more_clear:
                self.image_ids.append(the_id)
                cats = np.zeros(80)
                cats[[named_categories.index(lab) for lab in ds_to_coco[label]]] = 1
                self.imageid_to_label[the_id] = [cats, country, region, income]
        
        if self.setnum is not None:
            if 'incomeset' in self.setnum:
                incomes = np.array([round(np.log(float(self.imageid_to_label[imageid][3]))/3) for imageid in self.image_ids]) ## are all 1-3
                if self.setnum[-8:] == "-income2":
                    setind = int(self.setnum[9:-8])
                else:
                    setind = int(self.setnum[9:])
                incomes1 = np.where(incomes==1)[0]
                incomes3 = np.where(incomes==3)[0]
                min_len = np.amin([len(incomes1), len(incomes3)])
                keep13 = np.concatenate([np.random.choice(incomes1, size=int((10-setind)*.1*min_len), replace=False), np.random.choice(incomes1, size=int(setind*.1*min_len), replace=False)])
                if self.setnum[-8:] == "-income2":
                    keep2 = np.where(incomes!=2)[0]
                else:
                    keep13 = np.concatenate([np.where(incomes==2)[0], keep13])
                self.image_ids = np.array(self.image_ids)[keep13] 

            elif 'income' in self.setnum:
                ind = int(self.setnum[6:])
                self.image_ids = [imageid for imageid in self.image_ids if round(np.log(float(self.imageid_to_label[imageid][3]))/3) == ind]
            else:
                assert NotImplementedError
        self.image_ids = np.array(self.image_ids)

class FairFaceDataset(data.Dataset):
    
    def __init__(self, transform, version = '', only=None, setnum=None): 
        self.version = version # train or val
        self.split = version.split('_')[0]
        self.att = '_'.join(version.split('_')[1:])

        self.races = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
        self.genders = ['Female', 'Male']
        self.celeba = ['Arched_Eyebrows', 'Attractive', 'Bushy_Eyebrows', 'Pointy_Nose', 'Receding_Hairline', 'Young', 'Gray_Hair', 'High_Cheekbones', 'Blond_Hair', 'Bags_Under_Eyes', 'Wearing_Earrings', 'Mouth_Slightly_Open', 'Eyeglasses', 'Wearing_Hat', 'Black_Hair', 'Brown_Hair', 'Bangs', 'Smiling']
        self.only = only
        self.setnum = setnum

        self.transform = transform
        self.img_folder = '/Data/FairFace/' + self.split

        self.setup_anns()


    def __getitem__(self, index):
        image_id = self.image_ids[index]

        file_path = os.path.join(self.img_folder, image_id) + '.jpg'

        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[:-4]
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)

        anns = self.imageid_to_label[image_id]
        if self.att in self.races:
            target = np.array([anns[1]==self.att])
        elif self.att in self.genders:
            target = np.array([anns[2]==self.att])
        elif self.att == 'age':
            young = ['0-2', '3-9', '10-19', '20-29', '30-39']
            target = np.array([anns[3] in young])
        elif self.att in self.celeba:
            target = np.array([anns[4]])
        else:
            assert NotImplementedError
        if self.split == 'val':
            return image, target, [anns[1] == 'White', anns[2] == 'Female']
        return image, target

    def setup_anns(self):
        annotations = pd.read_csv('/Data/FairFace/fairface_label_{}.csv'.format(self.split))

        self.image_ids = []
        self.imageid_to_label = {}
        if self.att in self.celeba:
            if self.only is None:
                loss_info = pickle.load(open('models/modeloresnet50_weightsopytorch_imagenet1k_v2_dataoceleba_{}_traino-1/dupo0/loss_info.pkl'.format(self.att), 'rb'))
                probs = np.array(loss_info['ff_probs'][-1])

                celeba_annotations = pd.read_csv('/Data/CelebA/celeba/list_attr_celeba.txt', delim_whitespace=True, header=1)
                celeba_annots = celeba_annotations[self.att]
                celeba_annots[celeba_annots==-1]=0
                mean = np.mean(celeba_annots)/np.mean(probs)
                probs = np.clip(probs*mean, 0, 1)
            else:
                loss_info = pickle.load(open('models/modeloresnet50_weightsopytorch_imagenet_dataoceleba_{0}-{1}_traino-1/dupo0/loss_info.pkl'.format(self.att, self.only), 'rb'))
                probs = np.array(loss_info['ff_probs'][-1])
       
        for i in range(len(annotations)):
            annot = annotations.iloc[i]
            imageid = os.path.basename(annot['file'])[:-4]
            if self.only is not None:
                if annot['gender'] != self.only:
                    continue
            self.image_ids.append(imageid)
            self.imageid_to_label[imageid] = [annot['file'], annot['race'], annot['gender'], annot['age'], None]
            if self.att in self.celeba:
                if self.split == 'train':
                    self.imageid_to_label[imageid][4] = probs[i][0]
                else:
                    self.imageid_to_label[imageid][4] = np.random.randint(0, 2)

        self.image_ids = np.array(self.image_ids)

        if self.setnum is not None:
            if self.setnum == 'set10':
                incr_sets = pickle.load(open('fairface/{}_incr_splits_with11.pkl'.format(self.att), 'rb')) 
                self.image_ids = self.image_ids[incr_sets[-1]]
            else:
                incr_sets = pickle.load(open('fairface/{}_incr_splits.pkl'.format(self.att), 'rb')) 
                self.image_ids = self.image_ids[incr_sets[int(self.setnum[3:])]]


class CelebADataset(data.Dataset):
    
    def __init__(self, transform, version = '', only=None, setnum=None, trainnum=-1): 
        self.version = version # train or val
        split = version.split('_')[0]
        self.att = '_'.join(version.split('_')[1:])
        self.trainnum = trainnum
        self.setnum = setnum
        if '+' in self.att:
            self.att1, self.att2 = self.att.split('+')
        self.only = only
        self.skew = 0
        if self.att[-5:] == '_skew':
            self.att = self.att[:-5]
            self.skew = 1
        elif self.att[-6:] == '_skewz':
            self.att = self.att[:-6]
            self.skew = 2
        elif self.att[-6:] == '_skewa': # no skew essentially
            self.att = self.att[:-6]
            self.skew = 3
        if split == 'train':
            self.version_ref = 0
        elif split == 'val':
            self.version_ref = 1
        else:
            assert NotImplementedError
            
        self.rotation = (self.att == 'Rotate')
        if self.rotation:
            assert self.skew == 0

        self.transform = transform
        self.img_folder = '/Data/CelebA/celeba/img_align_celeba/'
        self.setup_anns()


    def __getitem__(self, index):
        image_id = self.image_ids[index]

        file_path = os.path.join(self.img_folder, image_id) + '.jpg'

        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[:-4]
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)

        anns = self.imageid_to_label[image_id]
        target = np.array([anns])
        if self.rotation:
            image = torch.rot90(image, anns, dims=[1, 2])
            target = np.zeros(4)
            target[anns] = 1

        if self.version_ref == 1:
            return image, target, self.imageid_to_att[image_id]

        return image, target

    def setup_anns(self):
        annotations = pd.read_csv('/Data/CelebA/celeba/list_attr_celeba.txt', delim_whitespace=True, header=1)
        filenames = pd.read_csv('/Data/CelebA/celeba/list_eval_partition.txt', delim_whitespace=True, header=None)
        self.categories = np.sort(list(annotations.keys()))
        assert len(annotations) == len(filenames)

        self.image_ids = []
        self.imageid_to_label = {}
        self.imageid_to_att = {}

        if self.only is not None:
            all_labs = []
        
        for i in range(len(annotations)):
            if filenames[1].iloc[i] == self.version_ref:
                imageid = filenames[0].iloc[i][:-4]
                annots = annotations.iloc[i]
                if self.only is not None: # only means only that group, btw
                    if self.only == 'Female':
                        all_labs.append(annots['Male']==1)
                    else:
                        all_labs.append(annots[self.only]==1)

                    if self.only == 'Female':
                        if annots['Male'] == 1:
                            continue
                    elif annots[self.only] != 1:
                        continue
                self.image_ids.append(imageid)
                if self.rotation:
                    self.imageid_to_label[imageid] = np.random.randint(0, 4)
                else:
                    if '+' in self.att:
                        self.imageid_to_label[imageid] = (annots[self.att1]==1 or annots[self.att2]==1)
                    else:
                        self.imageid_to_label[imageid] = annots[self.att]==1

                if '+' in self.att:
                    self.imageid_to_att[imageid] = [annots[self.att1]==1, annots[self.att2]==1]
                else:
                    self.imageid_to_att[imageid] = [annots['Young']==1, annots['Male']==-1]
        self.image_ids = np.array(self.image_ids)
        if self.skew > 0 and self.version_ref == 0: # .5832 of positive and negatives labels are going to be female
            assert self.only is None
            f = set(np.where([self.imageid_to_att[imageid][1] for imageid in self.image_ids])[0])
            m = set(np.arange(len(self.image_ids))).difference(f)
            att = set(np.where([self.imageid_to_label[imageid] for imageid in self.image_ids])[0])
            noatt = set(np.arange(len(self.image_ids))).difference(att)

            if self.skew == 1:
                mult = 1.3994954639126418
            elif self.skew == 2:
                mult = 4
            elif self.skew == 3:
                mult = 1
            all_imageids = set()
            for t, this_att in enumerate([att, noatt]):
                if t == 0:
                    f_thisatt = f&this_att
                    m_thisatt = m&this_att
                elif t == 1:
                    f_thisatt = m&this_att
                    m_thisatt = f&this_att
                if len(m_thisatt)*mult < len(f_thisatt):
                    all_imageids = all_imageids.union(np.random.choice(list(f_thisatt), size=int(len(m_thisatt)*mult), replace=False))
                    all_imageids = all_imageids.union(m_thisatt)
                else:
                    all_imageids = all_imageids.union(np.random.choice(list(m_thisatt), size=int(len(f_thisatt)/mult), replace=False))
                    all_imageids = all_imageids.union(f_thisatt)
            f_att = f&att
            f_noatt = f&noatt
            m_att = m&att
            m_noatt = m&noatt

            self.image_ids = self.image_ids[np.array(list(all_imageids))]
        if self.only is not None:
            _, counts = np.unique(all_labs, return_counts=True)
            min_len = min(counts)
            self.image_ids = np.random.choice(self.image_ids, size=min_len, replace=False)

        if self.setnum is not None:
            if '+' not in self.att:
                incr_sets = pickle.load(open('dataset_files/{}_incr_splits.pkl'.format(self.att), 'rb')) 
                self.image_ids = self.image_ids[incr_sets[int(self.setnum[3:])-1]] ## WHOOPS, -1 error that I have to keep now
            else:
                if self.trainnum == -1:
                    self.trainnum = len(self.image_ids)
                all_atts = [self.imageid_to_att[imageid] for imageid in self.image_ids]
                all_labs = [self.imageid_to_label[imageid] for imageid in self.image_ids]
                all_atts = np.array(all_atts)
                pos_prop = np.mean(all_labs)
                if '-prop' in self.setnum:
                    self.setnum, self.prop = self.setnum.split('-')
                    pos_prop = (int(self.prop[4:]))*.1
                att1_prop, att2_prop = np.mean(all_atts[:, 0]), np.mean(all_atts[:, 1])
                set_num = int(self.setnum[3:])
                # the increments where set0 is all att1, and set10 is all att2, so set5 is 50-50
                num_att1 = int(self.trainnum*(pos_prop*(10-set_num)*.1))
                num_att2 = int(self.trainnum*(pos_prop*(set_num)*.1))

                att1_spots = np.where(np.logical_and(all_atts[:, 0]==1, all_atts[:, 1]==0))[0]
                att2_spots = np.where(np.logical_and(all_atts[:, 1]==1, all_atts[:, 0]==0))[0]
                neg_spots = np.where(np.logical_and(all_atts[:, 0]==0, all_atts[:, 1]==0))[0]

                fact = 1.
                num_neg = int(fact*self.trainnum)-num_att1-num_att2
                while num_att1 > len(att1_spots) or num_att2 > len(att2_spots) or num_neg > len(neg_spots):
                    num_att1, num_att2 = int(.9*num_att1), int(.9*num_att2)
                    fact *= .9
                    num_neg = int(fact*self.trainnum)-num_att1-num_att2
               
                pos_samps = np.concatenate([np.random.choice(att1_spots, size=num_att1, replace=False), np.random.choice(att2_spots, num_att2, replace=False)])
                neg_samps = np.random.choice(neg_spots, size=int(fact*self.trainnum) - len(pos_samps), replace=False)
                self.image_ids = self.image_ids[np.concatenate([pos_samps, neg_samps])]


class CelebASaliencyDataset(data.Dataset):
    
    def __init__(self, transform, version = ''): 
        self.version = version # train or val
        split = version.split('_')[0]
        self.atts = '_'.join(version.split('_')[1:])
        self.inverse = False
        if len(self.atts) == 3:
            self.inverse = True
        self.att1, self.att2 = self.atts.split('&')[:2]

        if split == 'train':
            self.version_ref = 0
        elif split == 'val':
            self.version_ref = 1
        else:
            assert NotImplementedError

        self.transform = transform
        self.img_folder = '/Data/CelebA/celeba/img_align_celeba/'
        self.setup_anns()


    def __getitem__(self, index):
        image_id = self.image_ids[index]

        file_path = os.path.join(self.img_folder, image_id) + '.jpg'

        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[:-4]
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)

        anns = self.imageid_to_label[image_id]
        target = np.array([anns])
        if self.version_ref == 1:
            return image, target, self.imageid_to_att[image_id]

        return image, target

    def setup_anns(self):
        annotations = pd.read_csv('/Data/CelebA/celeba/list_attr_celeba.txt', delim_whitespace=True, header=1)
        filenames = pd.read_csv('/Data/CelebA/celeba/list_eval_partition.txt', delim_whitespace=True, header=None)
        self.categories = np.sort(list(annotations.keys()))
        assert len(annotations) == len(filenames)

        self.image_ids = []
        self.imageid_to_label = {}
        self.imageid_to_att = {}

        if self.version_ref == 1: 
            counts = [[] for _ in range(4)]
        
        for i in range(len(annotations)):
            if filenames[1].iloc[i] == self.version_ref:
                imageid = filenames[0].iloc[i][:-4]
                annots = annotations.iloc[i]
                if self.version_ref == 0:
                    if self.inverse:
                        if annots[self.att1] != annots[self.att2]:
                            self.image_ids.append(imageid)
                            self.imageid_to_label[imageid] = annots[self.att1]==1
                    else:
                        if annots[self.att1] == annots[self.att2]:
                            self.image_ids.append(imageid)
                            self.imageid_to_label[imageid] = annots[self.att1]==1

                elif self.version_ref == 1: # 4 equally constructed quadrants
                    if annots[self.att1] == 1:
                        if annots[self.att2] == 1:
                            counts[0].append(imageid)
                        else:
                            counts[1].append(imageid)
                    else:
                        if annots[self.att2] == 1:
                            counts[2].append(imageid)
                        else:
                            counts[3].append(imageid)
                    self.imageid_to_att[imageid] = [annots[self.att1]==1, annots[self.att2]==1]
                    self.imageid_to_label[imageid] = annots[self.att1]==1
        
        if self.version_ref == 0:
            labels = np.array([self.imageid_to_label[imageid] for imageid in self.image_ids])
            neg = np.where(labels==0)[0]
            pos = np.where(labels==1)[0]
            min_len = min(len(neg), len(pos))
            self.image_ids = np.array(self.image_ids)
            self.image_ids = np.concatenate([np.random.choice(self.image_ids[pos], size=min_len, replace=False), np.random.choice(self.image_ids[neg], size=min_len, replace=False)])

            ## sanity check ###
            labels = np.array([self.imageid_to_label[imageid] for imageid in self.image_ids])
            neg = np.where(labels==0)[0]
            pos = np.where(labels==1)[0]
            assert len(neg) == len(pos)
        elif self.version_ref == 1:
            min_count = np.amin([len(chunk) for chunk in counts])
            for chunk in range(len(counts)):
                self.image_ids.extend(np.random.choice(counts[chunk], size=min_count, replace=False))

        self.image_ids = np.array(self.image_ids)

class GeodeDataset(data.Dataset):
    
    def __init__(self, transform, version = ''): # version can be 'train' or 'test'
        # test and test_with_gen are same images, just with gender or not

        self.annotations = pd.read_csv('/Data/geode/index.csv')
        self.categories = np.sort(np.unique(np.array(self.annotations['object'])))

        self.version = version
        if not os.path.exists('geode_splits.pkl'):
            train_indices, test_indices = train_test_split(np.arange(len(self.annotations)), test_size=0.2)
            pickle.dump([train_indices, test_indices], open('geode_splits.pkl', 'wb'))
        else:
            train_indices, test_indices = pickle.load(open('geode_splits.pkl', 'rb'))

        if self.version == 'train':
            self.image_ids = train_indices
        elif self.version == 'test':
            self.image_ids = test_indices
        else:
            assert False

        self.transform = transform
        self.img_folder = '/Data/geode/images'


    def __getitem__(self, index):
        image_id = self.image_ids[index]

        file_path = os.path.join(self.img_folder, self.annotations.iloc[image_id]['file_path'])
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)

        target = np.zeros(len(self.categories))
        target[list(self.categories).index(self.annotations.iloc[image_id]['object'])] = 1
        if self.version == 'train':
            return image, target
        elif self.version == 'test':
            return image, target, self.annotations.iloc[image_id]['region']


    def __len__(self):
        return len(self.image_ids)

def read_xml_content(xml_file):
    parser = etree.XMLParser(recover=True)
    tree = ET.parse(xml_file, parser=parser)
    root = tree.getroot()

    list_with_all_boxes = []
    filename = root.find('filename').text
    width, height = float(root.find('size').find('width').text), float(root.find('size').find('height').text) # x is width
    instances = []

    for boxes in root.iter('object'):

        instance = boxes.find('name').text.strip().lower()
        instance = ' '.join(instance.split())
        instance = instance.replace('occluded', '').replace('crop', '').strip()
        instances.append(instance)
    return instances

class ImageNetDataset(data.Dataset):

    def __init__(self, transform, version):
        self.transform = transform
        self.version = version
        
        self.img_folder = '/Data/ImageNet/ILSVRC_2014_Images/ILSVRC2014_DET_train'
        self.annotations_folder = '/Data/ImageNet/ILSVRC_2014_Annotations/ILSVRC2014_DET_bbox_train'
        self.image_ids = [str(num).zfill(8) for num in range(1, 60659)]

        if not os.path.exists('imagenet_splits.pkl'):
            train_indices, test_indices = train_test_split(self.image_ids, test_size=0.2)
            pickle.dump([train_indices, test_indices], open('imagenet_splits.pkl', 'wb'))
        else:
            train_indices, test_indices = pickle.load(open('imagenet_splits.pkl', 'rb'))

        if self.version == 'train':
            self.image_ids = train_indices
        elif self.version == 'test':
            self.image_ids = test_indices
        else:
            assert False

        self.image_ids = np.array(self.image_ids)

        meta = loadmat('/Data/ImageNet/ILSVRC_2014_Devkit/ILSVRC2014_devkit/data/meta_det.mat')['synsets'][0]
        self.labels_to_names = {chunk[1][0]: chunk[2][0] for chunk in meta if chunk[0][0] < 201}

        self.categories = list(np.sort(list(self.labels_to_names.keys())))
        
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_path = self.img_folder + '/ILSVRC2014_train_' + image_id[:4] + '/ILSVRC2014_train_' + image_id + '.JPEG'
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[17:-5]

        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)
        image_size = list(image.size())[1:]

        formatted_anns = read_xml_content(self.annotations_folder + '/ILSVRC2014_train_' + image_id[:4] + '/ILSVRC2014_train_' + image_id + '.xml')

        target = np.zeros(len(self.categories))
        for instance in formatted_anns:
            target[self.categories.index(instance)] = 1

        if self.version == 'train':
            return image, target
        elif self.version == 'test':
            return image, target, np.array([0])





