#!/usr/bin/env python
# coding:utf-8
"""
@author: zc12345 
@contact: 18292885866@163.com

@file: view_label.py
@time: 2019/7/16 20:20
@description: polygon label to segmentation

"""
import base64
import json
import os
import os.path as osp

import PIL.Image
import yaml

from labelme.logger import logger
from labelme import utils


def json2dataset(json_file, out=None):
    if out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

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

    label_name_to_value = {'_background_': 0, 'road':1, 'car':2, 'motocycle':3, 'motorcycle':3, 'person':3}
    #for shape in sorted(data['shapes'], key=lambda x: x['label']):
    #    label_name = shape['label']
    #    if label_name in label_name_to_value:
    #        label_value = label_name_to_value[label_name]
    #    else:
    #        label_value = len(label_name_to_value)
    #        label_name_to_value[label_name] = label_value
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    # lbl_viz = utils.draw_label(lbl, img, label_names)

    # PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    out_label_png = osp.join(out_dir, file_name+'_label.png')
    utils.lblsave(out_label_png, lbl)
    # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, file_name+'_label_viz.png'))

    logger.info('Saved to: {}'.format(out_label_png))


def dir_process(src_dir):
    img_path = 'img'
    label_path = 'label'
    vis_path = 'vis'
    imgs_dir = os.path.join(src_dir, img_path)
    labels_dir = os.path.join(src_dir, label_path)
    vis_dir = os.path.join(src_dir, vis_path)
    if not osp.exists(labels_dir):
        return
    for label_file in os.listdir(labels_dir):
        if label_file.split('.')[-1] == 'json':
            json_path = os.path.join(labels_dir, label_file)
            print(json_path)
            json2dataset(json_path, vis_dir)

def main():
    src = './'
    for root, dirs, _ in os.walk(src):
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            dir_process(src_dir)

if __name__ == '__main__':
    #dir_process('sequence-17')
    main()
