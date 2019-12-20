#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./logs.log',
                    filemode='w')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-k', help="input the keypoint number on one side", default=8, type=int)
args = parser.parse_args()
K = args.k

class genearte_list:
    def __init__(self, data_dir, err_lst_path, target_train_path, target_test_path):
        self.data_dir = data_dir
        self.target_train_path = osp.join(data_dir, target_train_path)
        self.target_test_path = osp.join(data_dir,target_test_path)
        self.err_lst_path = osp.join(data_dir,err_lst_path)

    def get_list(self):
        data = []
        logging.info(self.data_dir)
        err_lst = []
        with open(self.err_lst_path, 'r') as err_f:
            for line in err_f.readlines():
                err_lst.append(line.strip())
        for root, dirs, files in os.walk(self.data_dir): # data dir
            for dir_ in dirs:
                # img_dir = osp.join(root, dir_, 'img')
                # label_dir = osp.join(root, dir_, 'vis')
                anno_dir = osp.join(root, dir_, 'label')
                for _, _, annos in os.walk(anno_dir):
                    for anno_file in annos:
                        if anno_file.split('.')[0] in err_lst:
                            logging.info('File {} in ERROR_LIST.'.format(anno_file))
                            continue
                        if anno_file.split('.')[-1] == 'json':
                            fn = anno_file.split('.')[0]
                            img = osp.join(dir_, 'img', fn + '.jpg').replace('\\', '/')
                            json_lbl = osp.join(dir_, 'label', fn + '.json').replace('\\', '/')
                            t = (img, json_lbl)
                            img_path, json_lbl_path = \
                                            osp.join(self.data_dir, img), osp.join(self.data_dir, json_lbl)
                            assert osp.exists(img_path) and osp.exists(json_lbl_path) #and osp.exists(seg_lbl_path), 'image-label file not found'
                            data.append(t)
        return data
    
    def _single_get_list(self, dir_path):
        # remove in final version
        data = []
        anno_dir = osp.join(dir_path, 'label')
        for _, _, annos in os.walk(anno_dir):
            for anno_file in annos:
                if anno_file.split('.')[-1] == 'json':
                    fn = anno_file.split('.')[0]
                    img = osp.join(dir_path, 'img', fn + '.jpg').replace('\\', '/')
                    seg_lbl = osp.join(dir_path, 'vis', fn + '_label.png').replace('\\', '/')
                    json_lbl = osp.join(dir_path, 'label', fn + '.json').replace('\\', '/')
                    t = (img, json_lbl)
                    img_path, json_lbl_path = \
                                    osp.join(self.data_dir, img), osp.join(self.data_dir, json_lbl)
                    assert osp.exists(img_path) and osp.exists(json_lbl_path) #and osp.exists(seg_lbl_path), 'image-label file not found'
                    data.append(t)
        return data
    
    def _single_write(self, dir_path):
        # remove in final version
        data = self._single_get_list(dir_path)
        train_data, test_data = self.split_data(data, 0.2)
        with open(self.target_train_path, 'w') as f:
            for line in train_data:
                img, label = line
                f.write('{} {}\n'.format(img, label))
        with open(self.target_test_path, 'w') as f:
            for line in test_data:
                img, label = line
                f.write('{} {}\n'.format(img, label))
        
    def split_data(self, data, test_ratio):
        train, test = [], []
        np.random.seed(42)
        shuffle_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffle_indices[:test_set_size]
        train_indices = shuffle_indices[test_set_size:]
        for i in train_indices:
            train.append(data[i])
        for i in test_indices:
            test.append(data[i])
        return train, test
    
    def write_list(self):
        data = self.get_list()
        train_data, test_data = self.split_data(data, 0.2)
        with open(self.target_train_path, 'w') as f:
            for line in train_data:
                img, label = line
                f.write('{} {}\n'.format(img, label))
        with open(self.target_test_path, 'w') as f:
            for line in test_data:
                img, label = line
                f.write('{} {}\n'.format(img, label))


if __name__ == '__main__':
    data_dir = '../tsd-max-traffic/'
    err_lst_path = 'error_lst.txt'
    target_train_path = 'train_lst.txt'
    target_test_path = 'test_lst.txt'
    g = genearte_list(data_dir, err_lst_path, target_train_path, target_test_path)
    g.write_list()


