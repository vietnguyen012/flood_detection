# uncompyle6 version 3.9.0a1
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.7.0 (default, Oct  9 2018, 10:31:47) 
# [GCC 7.3.0]
# Embedded file name: /root/flood/data.py
# Compiled at: 2022-02-09 11:01:06
# Size of source mod 2**32: 7117 bytes
import pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os, pandas as pd
from os import listdir
import json
from transformers import AutoTokenizer

class TextImageDataset(Dataset):

    def __init__(self, data_dir, img_dir, transform, label_file, text_file, language_pretraind_model, max_length=384, is_test=False):
        self.language_pretrained_model = language_pretraind_model
        self.img_path = os.path.join(data_dir, img_dir)
        self.data_dir = data_dir
        self.img_dir = img_dir
        text_path = os.path.join(data_dir, text_file)
        self.is_test = is_test
        self.image_files = [f for f in listdir(self.img_path)]
        if not self.is_test:
            positive = 0
            negative = 0
            label_path = os.path.join(data_dir, label_file)
            label_id_frame = pd.read_csv(label_path, header=None)
            image_id = label_id_frame[0].tolist()
            labels = label_id_frame[1].tolist()
            self.id2label = {}
            for id, label in zip(image_id, labels):
                self.id2label[int(id)] = int(label)
                if int(label) == 1:
                    positive += 1
                else:
                    negative += 1

            self.pos_weight = positive / negative
        self.text_list = json.load(open(text_path))['images']
        self.tokenizer = AutoTokenizer.from_pretrained(language_pretraind_model)
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_file = os.path.join(self.data_dir, os.path.join(self.img_dir, self.image_files[index]))
        id_img = img_file.split('/')[(-1)].split('_')[0].split(".")[0]
        img = Image.open(img_file).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.is_test:
            label = self.id2label[int(id_img)]

        text = ''
        if self.language_pretrained_model == 'roberta-base':
            sep = ' </s> '
        else:
            sep = ' [SEP] '

        for item in self.text_list:
            if item['image_id'] == id_img:
                try:
                    title = '' if item['title'] == None else ' '.join(item['title'].split())
                    desc = '' if (item['description'] == 'null' or item['description'] == None) else (' '.join(item['description'].split()))
                    user_tag = sep.join(item['user_tags'])
                    text = title + sep + desc + sep + user_tag
                    if text == '':
                        raise ValueError('empty text')
                        print()
                except:
                    print()

        token = self.tokenizer(text, max_length=(self.max_length), truncation=True, padding='max_length')
        if not self.is_test:
            return {'id':id_img,  'img_tensor':img,
             'input_ids':torch.LongTensor(token['input_ids']),
             'attention_mask':torch.LongTensor(token['attention_mask']),
             'label':torch.FloatTensor([label])}
        else:
            return {'id': id_img, 'img_tensor': img,
                    'input_ids': torch.LongTensor(token['input_ids']),
                    'attention_mask': torch.LongTensor(token['attention_mask']),
                    }

class TextDataset(Dataset):

    def __init__(self, data_dir, img_dir, transform, label_file, text_file, language_pretraind_model, max_length=384, is_test=False, is_dev=False, metric_learning=False):
        self.is_dev = is_dev
        self.language_pretrained_model = language_pretraind_model
        self.img_path = os.path.join(data_dir, img_dir)
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.metric_learning = metric_learning
        text_path = os.path.join(data_dir, text_file)
        self.is_test = is_test
        self.image_files = [f for f in listdir(self.img_path)]
        if (self.is_test or self).is_dev:
            label_path = os.path.join(data_dir, label_file)
            label_id_frame = pd.read_csv(label_path)
            image_id = label_id_frame['orig_id'].tolist()
            image_ref_id = label_id_frame['ref_id'].tolist()
            labels = label_id_frame['label '].tolist()
            self.id2label = {}
            for id, ref_id, label in zip(image_id, image_ref_id, labels):
                self.id2label[int(ref_id)] = (
                 int(label), id)

        else:
            positive = 0
            negative = 0
            label_path = os.path.join(data_dir, label_file)
            label_id_frame = pd.read_csv(label_path, header=None)
            image_id = label_id_frame[0].tolist()
            labels = label_id_frame[1].tolist()
            self.id2label = {}
            for id, label in zip(image_id, labels):
                self.id2label[int(id)] = int(label)
                if int(label) == 1:
                    positive += 1
                else:
                    negative += 1

            self.pos_weight = positive / negative

        self.text_list = json.load(open(text_path))['images']
        self.tokenizer = AutoTokenizer.from_pretrained(language_pretraind_model)
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_file = os.path.join(self.data_dir, os.path.join(self.img_dir, self.image_files[index]))
        if self.is_dev:
            id_img = img_file.split('/')[(-1)].split('.')[0]
        else:
            id_img = img_file.split('/')[(-1)].split('_')[0]
        if not self.is_test:
            if self.is_dev:
                label = self.id2label[int(id_img)][0]
            else:
                label = self.id2label[int(id_img)]

        text = ''
        if self.is_dev:
            orig_id_img = self.id2label[int(id_img)][1].split('_')[0]
        if self.language_pretrained_model == 'roberta-base':
            sep = ' </s> '
        else:
            sep = ' [SEP] '

        for item in self.text_list:
            if self.is_dev:
                if item['image_id'] == orig_id_img:
                    try:
                        title = '' if item['title'] == None else ' '.join(item['title'].split())
                        desc = '' if (item['description'] == 'null' or item['description'] == None) else (' '.join(item['description'].split()))
                        user_tag = sep.join(item['user_tags'])
                        text = title + sep + desc + sep + user_tag
                    except:
                        print()

                elif item['image_id'] == id_img:
                    try:
                        title = '' if item['title'] == None else ' '.join(item['title'].split())
                        desc = '' if (item['description'] == 'null' or item['description'] == None) else (' '.join(item['description'].split()))
                        user_tag = sep.join(item['user_tags'])
                        text = title + sep + desc + sep + user_tag
                        if text == '':
                            raise ValueError('empty text')
                            print()
                    except:
                        print()

        token = self.tokenizer(text, max_length=(self.max_length), truncation=True, padding='max_length')
        if not self.is_test:
            return {
                 'input_ids':torch.LongTensor(token['input_ids']),
                 'attention_mask':torch.LongTensor(token['attention_mask']),
                 'label':torch.FloatTensor([label])}
        else:
            return {
                    'input_ids': torch.LongTensor(token['input_ids']),
                    'attention_mask': torch.LongTensor(token['attention_mask']),
                    }
if __name__ == '__main__':
   pass
