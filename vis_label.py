#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-04 16:30:21
# @Author  : zc12345 (18292885866@163.com)
# @Link    : https://github.com/zc12345/
# @Version : $Id$

import os.path as osp
import os
import json
import logging
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG,
                    filename='debug.log',
                    format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')

def plot_label(img_path, label_path, save_path):
    pts = load_pts(label_path)
    plt.figure(0)
    img = Image.open(img_path)
    w, h = img.size
    plt.imshow(img)
    plt.plot(pts[:,0], pts[:, 1], '-*')
    for i, (x, y) in enumerate(zip(pts[:,0],pts[:,1])):
        plt.text(x, y+1, str(i), ha = 'center',va = 'bottom',fontsize=20)
        plt.plot(pts[:,0],pts[:,1],
             markersize=10, marker='o', color='red', markerfacecolor='blue',
             linestyle='dashed', linewidth=3, markeredgecolor='m')
    fig = plt.gcf()
    # fig.set_size_inches(w / 100.0, h / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, format='png', transparent=True)
    plt.close(0)
    plt.show()


def load_pts(file_path):
    ext = file_path.split('.')[-1]
    points = None
    if ext == 'json':
        data = json.load(open(file_path))
        if 'points' in data.keys():
            points = np.array(data['points'])
        elif 'version' in data.keys():
            for shape in data['shapes']:
                if shape['label'] == 'road':
                    points = np.array(shape['points'])
    elif ext == 'txt':
        points = np.loadtxt(file_path)
    else:
        logging.error('{} Unsolved File Extension: {}.'.format(file_path, ext))
    return points

def main():
    data_dir = '../../data/tsd-max-traffic'
    for k in range(2,9):
        val_lst_path = os.path.join(data_dir, 'list/test_full_lst-{}.txt'.format(k))
        train_lst_path = os.path.join(data_dir, 'list/train_full_lst-{}.txt'.format(k))
        save_fig_dir = os.path.join(data_dir, 'kpts_lbl_{}'.format(k))
        if not os.path.exists(save_fig_dir):
            os.mkdir(save_fig_dir)
        with open(val_lst_path, 'r') as f:
            for line in f.readlines():
                img, seg, lbl, kpts_lbl = line.split()
                fn = img.split('/')[-1]
                fn = fn.split('.')[0]
                print('{} points, File {}'.format(k, img))
                img_path = os.path.join(data_dir, img)
                kpts_lbl_path = os.path.join(data_dir, kpts_lbl)
                lbl_path = os.path.join(data_dir, lbl)
                save_fig_path = os.path.join(save_fig_dir, fn + '.png')
                plot_label(img_path, kpts_lbl_path, save_fig_path)
                # plot_label(img_path, lbl_path, save_fig_path)
        with open(train_lst_path, 'r') as f:
            for line in f.readlines():
                img, seg, lbl, kpts_lbl = line.split()
                fn = img.split('/')[-1]
                fn = fn.split('.')[0]
                print('{} points, File {}'.format(k, img))
                img_path = os.path.join(data_dir, img)
                kpts_lbl_path = os.path.join(data_dir, kpts_lbl)
                lbl_path = os.path.join(data_dir, lbl)
                save_fig_path = os.path.join(save_fig_dir, fn + '.png')
                plot_label(img_path, kpts_lbl_path, save_fig_path)
                # plot_label(img_path, lbl_path, save_fig_path)

if __name__ == '__main__':
    main()