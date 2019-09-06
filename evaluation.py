from PIL import Image, ImageDraw
import numpy as np
import os.path as osp
import os
import json
import logging
import matplotlib.pyplot as plt
import time

logging.basicConfig(level=logging.DEBUG,
                    filename='debug.log',
                    format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')

# set right bottom and left bottom coordinates
coord1, coord2, delta = (1279, 1023), (0, 1023), 5
h, w = 1024, 1280

def calc_iou(vertices1, vertices2, h ,w):
    '''
    https://github.com/AlexMa011/pytorch-polygon-rnn/blob/master/utils/utils.py
    calculate iou of two polygons
    :param vertices1: vertices of the first polygon
    :param vertices2: vertices of the second polygon
    :return: the iou, the intersection area, the union area
    '''
    img1 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
    mask1 = np.array(img1)
    img2 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img2).polygon(vertices2, outline=1, fill=1)
    mask2 = np.array(img2)
    # plt.figure()
    # plt.imshow(img1)
    # plt.figure()
    # plt.imshow(img2)
    # plt.show()
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    nu = np.sum(intersection)
    de = np.sum(union)
    if de!=0:
        return nu*1.0/de, nu, de
    else:
        return 0, nu, de

def l2dist1(pts1, pts2):
    assert pts1.shape == pts2.shape
    m, n = pts1.shape
    dist = np.linalg.norm(pts1 - pts2, axis=1, ord=2)
    # dist = np.sqrt((pts1[:,0]-pts2[:,0])**2 + (pts1[:,1]-pts2[:,1])**2)
    # dist = []
    # for pt1, pt2 in zip(pts1, pts2):
    #     tmp = pt1 - pt2
    #     tmp = np.sqrt(np.sum(tmp**2))
    #     dist.append(tmp) 
    return np.mean(dist)

def l2dist2(pts1, pts2):
    assert pts1.shape == pts2.shape
    m, n = pts1.shape
    pts1 = _interp(pts1)
    dist = dist1 = []
    for pt in pts2:
        d = np.min(np.linalg.norm(pts1 - pt, axis=1, ord=2))
        # d = np.min(np.sqrt((pts1[:,0]-pt[0])**2+(pts1[:,1]-pt[1])**2))
        dist.append(d)
    return np.mean(dist)

def _interp(pts1):
    _x1 = np.arange(pts1[0, 0], pts1[-1, 0] + 1, step=1)
    _y1 = np.interp(_x1, pts1[:, 0], pts1[:, 1])
    _x1, _y1 = np.trunc(_x1), np.trunc(_y1)
    pts1 = np.array([_x1, _y1]).T
    return pts1

def load_pts(file_path, type=None):
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
    if points.shape[0] == 8:
        points[0][0] = 0
    if points is not None and type is None:
        points = array2kpts(points)
    return points

def array2kpts(pts):
    pts = np.clip(pts, (0,0), coord1)
    np.place(pts[:,0], pts[:,0] < delta, 0)
    np.place(pts[:,0], coord1[0] - pts[:,0] < delta, coord1[0])
    np.place(pts[:,1], coord1[1] - pts[:,1] < delta, coord1[1])
    kpts = []
    for point in pts:
        kpts.append(tuple(point))
    return kpts

def test():
    data_dir = '../../data/tsd-max-traffic'
    # file_path1 = osp.join(data_dir, 'sequence-1', 'label', 'Section14CameraC_00736c.json')
    # file_path2 = osp.join(data_dir, 'val-7pts-epoch50', 'Section5CameraC_00208c.txt')
    # pts1, pts2 = load_pts(file_path1), load_pts(file_path2)
    # coord = list([coord1, coord2])
    # pts2 = pts2 + coord
    # iou, nu, de = calc_iou(pts1, pts2, h=1024, w=1280)
    # print(iou, nu, de)
    file_path1 = osp.join(data_dir, 'val-7pts-epoch50', 'Section14CameraC_00400c.txt')
    file_path2 = osp.join(data_dir, 'val-7pts-epoch50', 'Section14CameraC_00520c.txt')
    kpt_label_pts, pred_pts = load_pts(file_path1, type='np'), load_pts(file_path2, type='np')
    dist1 = l2dist1(kpt_label_pts, pred_pts)
    dist2 = l2dist2(kpt_label_pts, pred_pts)
    print(dist1, dist2)
            


def miou(data_dir, val_lst_path, pts_num, epoch_num):
    ious = []
    nus = des = 0
    with open(val_lst_path, 'r') as f:
        for line in f.readlines():
            img, _ = line.split()
            set_dir, _, img_file = img.split('/')
            fn = img_file.split('.')[0]
            seg_label_path = osp.join(data_dir, set_dir, 'label', fn + '.json')
            kpt_label_path = osp.join(data_dir, set_dir, 'keypoint_label-{}'.format(pts_num), fn + '.json')
            pred_path = osp.join(data_dir, 'val_pred', 'val-{}pts-epoch{}'.format(pts_num, epoch_num), fn + '.txt')
            seg_label_pts, kpt_label_pts, pred_pts = load_pts(seg_label_path), load_pts(kpt_label_path), load_pts(pred_path)
            coord = list([coord1, coord2])
            pred_pts = pred_pts + coord
            iou, nu, de = calc_iou(seg_label_pts, pred_pts, h, w)
            # iou, nu, de = calc_iou(kpt_label_pts, pred_pts, h, w)
            ious.append(iou)
            nus += nu
            des += de
    return nus/des * 100

def mdist(data_dir, val_lst_path, pts_num, epoch_num):
    dists = []
    with open(val_lst_path, 'r') as f:
        for line in f.readlines():
            img, _ = line.split()
            set_dir, _, img_file = img.split('/')
            fn = img_file.split('.')[0]
            kpt_label_path = osp.join(data_dir, set_dir, 'keypoint_label-{}'.format(pts_num), fn + '.json')
            pred_path = osp.join(data_dir, 'val_pred', 'val-{}pts-epoch{}'.format(pts_num, epoch_num), fn + '.txt')
            kpt_label_pts, pred_pts = load_pts(kpt_label_path, type='np'), load_pts(pred_path, type='np')
            dist = l2dist2(kpt_label_pts, pred_pts)
            dists.append(dist)
    return np.mean(dists)


def show():
    ious = np.loadtxt('ious.csv')
    # for k in range(7):
    #     plt.plot(range(50), ious[k,:], linewidth=1.5, linestyle="-", label='{} points'.format(k+2))
    #     plt.ylabel('IoU')
    #     plt.xlabel('epoch')
    #     plt.legend(loc='upper left', frameon=False)
    #     plt.savefig('ious_fig{}.png'.format(k))
    #     plt.show()
    plt.plot(range(2,9), ious[:, 49], linewidth=1.5, linestyle="-.")
    plt.ylabel('IoU')
    plt.xlabel('num')
    plt.savefig('ious_fig_epoch{}.png'.format(50))
    plt.show()

def main():
    time0 = time.time()
    data_dir = '/mnt/data/data/tsd-max-traffic'
    val_lst_file = 'list/test_lst-2.txt'
    val_lst_path = osp.join(data_dir, val_lst_file)
    ious = np.zeros((7, 50))
    #mds = np.zeros((7, 50))
    for i in range(7):
        for j in range(50):
            time_start = time.time()
            iou = miou(data_dir, val_lst_path, i+2, j+1)
            # md = mdist(data_dir, val_lst_path, i+2, j+1)
            t = time.time() - time_start
            logging.info('keypoints {} epoch {}: mIoU = {}, time cost = {:.2f} s'.format(i+2, j+1, iou, t))
            ious[i,j] = iou
            #logging.info('keypoints {} epoch {}: mDist = {}, time cost = {:.2f} s'.format(i+2, j+1, md, t))
            #print('keypoints {} epoch {}: mDist = {}, time cost = {:.2f} s'.format(i+2, j+1, md, t))
            #mds[i, j] = md
    np.savetxt('ious.csv', ious, fmt='%0.8f')
    #np.savetxt('mdists.csv', mds, fmt='%0.8f')
    t = time.time() - time0
    print('time cost = {:.2f} s'.format(t))
    for k in range(7):
        plt.plot(range(50), ious[k,:], linewidth=2.5, linestyle="-", label='{} points'.format(k+2))
        #plt.plot(range(50), mds[k,:], linewidth=2.5, linestyle="-", label='{} points'.format(k+2))
    plt.ylabel('IoU')
    #plt.ylabel('E-Dist')
    plt.xlabel('epoch')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig('ious_fig.png')
    #plt.savefig('mdists_fig.png')
    plt.show()

if __name__ == '__main__':
    main()