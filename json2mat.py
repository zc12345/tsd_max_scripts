# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:42:20 2019

@author: zc12345
@contact: 18292885866@163.com

@description:
"""
import os.path as osp
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm
from norm_road_label import normPointLabel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-k', help="input the keypoint number on one side", default=8, type=int)
args = parser.parse_args()
K = args.k

def _load_json(json_path):
    data = json.load(open(json_path))
    return data['points']
    
import base64
import os
from labelme.logger import logger
from labelme import utils

def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.int32)
        instance_names = ['_background_']
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        if not label in label_name_to_value:
            continue
        shape_type = shape.get('shape_type', None)
        if type == 'class':
            cls_name = label
        cls_id = label_name_to_value[cls_name]
        mask = utils.shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
    return cls
    
def json2dataset(json_file):
    file_name = os.path.basename(json_file)
    file_name = file_name.split('.')[0]
    data = json.load(open(json_file))
    imageData = data.get('imageData')

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    #label_name_to_value = {'_background_': 0, 'road':1, 'car':2, 'motocycle':3, 'motorcycle':3, 'person':3}
    label_name_to_value = {'_background_': 0, 'road':1}
    lbl = shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    return lbl

class json2mat:
    def __init__(self, root_dir, lst_path, save_path, origin_img_shape=np.array([1280, 1024]), resized_img_shape=np.array([256, 256]), k=16):
        self._root_dir = root_dir
        self._lst_path = lst_path
        self._save_path = save_path
        self._origin_shape = origin_img_shape
        self._resized_shape = resized_img_shape
        self._k = k
        self._json_paths = []
        
    def _load_json_path(self):
        _path = osp.join(self._root_dir, self._lst_path)
        with open(_path, 'r') as f:
            for line in f.readlines():
                img, json_lbl = line.split()
                img_path = osp.join(self._root_dir, img).replace('\\', '/')
                json_path = osp.join(self._root_dir, json_lbl).replace('\\', '/')
                p = (img_path, json_path)
                assert osp.exists(img_path) and osp.exists(json_path)
                self._json_paths.append(p)
        return self._json_paths
    
    def _check_pts(self, img, kpts):
        m, n = kpts.shape
        w, h = img.size
        assert m == self._k and n == 2
        assert np.max(kpts[:, 0]) < w and np.max(kpts[:, 1]) < h
        assert w == self._origin_shape[0] and h == self._origin_shape[1]
        
    def _check_new(self, img, kpts):
        m, n = kpts.shape
        w, h = img.size
        assert m == self._k and n == 2
        assert np.max(kpts[:, 0]) < w and np.max(kpts[:, 1]) < h
        assert w == self._resized_shape[0] and h == self._resized_shape[1]
    
    def _resize(self, img_path, json_path, kpts):
        img = Image.open(img_path)
        seg = json2dataset(json_path)
        seg = Image.fromarray(seg)
        self._check_pts(img, kpts)
        img = img.resize(self._resized_shape)
        seg = seg.resize(self._resized_shape)
        _s = self._origin_shape / self._resized_shape
        kpts = kpts / _s
        kpts[kpts > self._resized_shape[0]-1] = self._resized_shape[0]-1
        kpts[kpts < 0 ] = 0
        #plt.imshow(img)
        #plt.plot(kpts[:,0], kpts[:,1])
        #plt.show()
        #plt.imshow(seg)
        #plt.show()
        self._check_new(img, kpts)
        return img, seg, kpts
        
    def convert(self):
        json_paths = self._load_json_path()
        num = len(json_paths)
        fname = [''] * num
        imgs = np.zeros((num, self._resized_shape[0], self._resized_shape[1], 3))
        kpts = np.zeros((num, self._k, 3))
        segs = np.zeros((num, self._resized_shape[0], self._resized_shape[1]))
        pbar = tqdm(total=num)
        for i, (img_path, json_path) in enumerate(json_paths):
            npl = normPointLabel(k=K)
            _, kpt = npl.normLabel(json_path)
            img, seg, kpt = self._resize(img_path, json_path, kpt)
            fname[i] = img_path.split('/')[-1]
            imgs[i] = img
            segs[i] = seg
            for j, pt in enumerate(kpt):
                kpts[i,j,0] = pt[0]
                kpts[i,j,1] = pt[1]
                kpts[i,j,2] = 1.0
            pbar.update(1)
        pbar.close()
        print('save data ...')
        sio.savemat(self._save_path, {'fname': fname, 'imgPath': imgs, 'ptsAll': kpts, 'segGT': segs})
         
def main():
    root_dir = '../tsd-max-traffic'
    lst_path = 'test_lst.txt'
    save_path = 'test-{}.mat'.format(K)
    j2m = json2mat(root_dir, lst_path, save_path, k=K*2)
    j2m.convert()

if __name__ == '__main__':
    main()
