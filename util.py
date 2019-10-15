#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 01:47:17 2019

@author: wuwenjun
"""

from bag import Bag
from word import Word
from feature import calculate_feature, get_histogram, get_histogram_cluster
from cluster import predict_kmeans
import numpy as np
import os
import glob
import cv2
import math
import h5py
import pickle
import csv
import pandas as pd
from sklearn.model_selection import KFold


def get_feat_from_image(image_path, save_flag, word_size,
                        histogram_bin=64, image=None, save_path=None):
    # print(image)
    if image is None:
        image = cv2.imread(image_path)
        assert image is not None, "imread fail, check path"
        image = np.array(image, dtype=int)
    words = Word(image, size=word_size)
    result = np.zeros([words.length, 320])

    for word, i in words:
        # get filename without extension
        if save_path is not None:
            dname = os.path.dirname(save_path)
            base = os.path.basename(image_path)
            path_noextend = os.path.splitext(base)[0]
            filename = os.path.join(dname, path_noextend)
        else:
            filename = None
        feat = calculate_feature(word, idx=i, save=save_flag, path=filename)
        hist = get_histogram(feat, nbins=histogram_bin)
        result[i, :] = hist
    if save_path is not None:
        pickle.dump(result, open(save_path, 'wb'))
    return result


def get_hist_from_image(image_path, kmeans, hclusters, dict_size, word_size,
                        image=None):
    if image is None:
        feat_words = get_feat_from_image(image_path, False, word_size)
    else:
        feat_words = get_feat_from_image(None, False, word_size, image=image)
    cluster_words = predict_kmeans(feat_words, kmeans, h_cluster=hclusters)
    hist_bag = get_histogram_cluster(cluster_words, dict_size=dict_size)
    return hist_bag


def load_mat(filename):
    f = h5py.File(filename, 'r')
    keys = list(f.keys())
    im = np.array(f["I"])
    im = np.transpose(im, [2, 1, 0])
    im = np.flip(im, 2)
    return im, np.array(f["M"])


def preprocess_roi_csv(csv_file):
    header = None
    case=y=x=width=heigh=0

    result = {}
    with open(csv_file, newline='') as f:
        f.seek(0)
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 0:
                header = row
                assert 'Case ID' in header, "No matching column with name: Case ID"
                assert 'Y' in header, "No matching column with name: Y"
                assert 'X' in header, "No matching column with name: X"
                assert 'Width' in header, "No matching column with name: Width"
                assert 'Height' in header, "No matching column with name: Height"
                case = header.index('Case ID')
                y = header.index('Y')
                x = header.index('X')
                width = header.index('Width')
                height = header.index('Height')
            if i > 0:
                try:
                    Y = int(row[y])
                except ValueError:
                    Y = int(float(row[y]))
                try:
                    X = int(row[x])
                except ValueError:
                   X = int(float(row[x]))
                c = int(row[case])
                try:
                    w = int(row[width])
                except ValueError:
                    w = int(float(row[width]))
                try:
                    h = int(row[height])
                except ValueError:
                    h = int(float(row[height]))
                bbox = [Y, Y+h, X, X+w]
                bb_l = result.get(c)
                if bb_l is None:
                    result[c] = [bbox]
                else:
                    result[c].append(bbox)
            i += 1
    return result

def preprocess_wsi_size_csv(csv_file):
    f = pd.read_csv(csv_file)
    caseID = f['Case ID']
    H = f['H']
    W = f['W']

    result = {}
    i = 0
    while i < len(caseID):
        # row_up, row_down, col_left, col_right

        height = H[i]
        width = W[i]
        bb_l = result.get(caseID[i])
        if bb_l is None:
            result[int(caseID[i])] = [height, width]
        i += 1
    return result


def calculate_label_from_mask(mask, size=3600, overlap_pixel=2400):
    bags = Bag(mask, size=size, overlap_pixel=overlap_pixel)
    label = np.zeros(bags.length)
    for bag, i in bags:
        if (bag == 1).any():
            label[i] = 1
        else:
            label[i] = 0
    return label


def bound_box(idx, w, length, size, overlap_pixel):
    """
    Function that return the bounding box of a word given its index
    Args:
        ind: int, ind < number of words

    Returns:
        Bounding box(int[]): [h_low, h_high, w_low, w_high]
    """
    assert idx < length, "Index Out of Bound"
    num_bag_w = int((w - overlap_pixel) / (size - overlap_pixel))
    box_h = int(math.floor(idx / num_bag_w) * (size - overlap_pixel))
    box_w = int(idx % (num_bag_w) * (size - overlap_pixel))

    return [box_h, box_h + size, box_w, box_w + size]


def calculate_label_from_roi_bbox(roi_bbox, wsi_size, factor=1,
                                  size=3600, overlap_pixel=2400):

    bags = Bag(h=wsi_size[0], w=wsi_size[1],
               size=size, overlap_pixel=overlap_pixel,
               padded=True)

    h, w = bags.h, bags.w
    pos_ind = set()
    num_bag_w = int(math.floor((w - overlap_pixel) / (size - overlap_pixel)))
    num_bag_h = int(math.floor((h - overlap_pixel) / (size - overlap_pixel)))
    # Bounding box(int[]): [h_low, h_high, w_low, w_high]
    for h_low, h_high, w_low, w_high in roi_bbox:
        h_low += bags.top
        h_high += bags.top
        w_low += bags.left
        w_high += bags.left
        #if h_high >= h and w_high >= w:
        if h_high > h or w_high > w:
            print("Size incompatible for case: {}".format(self.caseID))
            print("Bounding box: {}, {}, {}, {}".format(h_low, h_high, w_low, w_high))
            print("WSI size: {}, {}". format(h, w))
            h_high = min(h, h_high)
            w_high = min(w, w_high)
        ind_w_low = int(max(math.floor((w_low - size) / (size - overlap_pixel) +
           1), 0))
        ind_w_high = int(min(max(math.floor(w_high / (size - overlap_pixel)),
           0), num_bag_w - 1))
        ind_h_low = int(max(math.floor((h_low - size) / (size - overlap_pixel) +
           1), 0))
        ind_h_high = int(min(max(math.floor(h_high / (size - overlap_pixel)),
            0), num_bag_h - 1))
        for i in range(h_low, h_high+1):
            pos_ind.update(range(ind_h_low * num_bag_w + ind_w_low,
               ind_h_high * num_bag_w + ind_w_high + 1))
    pos_ind = np.sort(list(pos_ind))
    result = np.zeros(len(bags))
    result[pos_ind] = 1
    return result



def biggest_bbox(bbox_list):
    row_low = 1000000000000
    row_high = -1
    col_low = 1000000000000
    col_high = -1
    for bbox in bbox_list:
        row_low = (bbox[0] if bbox[0] < row_low else row_low)
        row_high = (bbox[1] if bbox[1] > row_high else row_high)
        col_low = (bbox[2] if bbox[2] < col_low else col_low)
        col_high = (bbox[3] if bbox[3] > col_high else col_high)
    return [row_low, row_high, col_low, col_high]


def crop_saveroi_batch(image_folder, dict_bbox, appendix='.jpg'):
    f_ls = glob.glob(os.path.join(image_folder, '*.tif'))
    for f in f_ls:
        base = os.path.basename(f)
        name_noextend = os.path.splitext(base)[0]
        outname = os.path.join(image_folder, 'roi', name_noextend + appendix)
        if not os.path.exists(outname):
            caseID = int(name_noextend.split('_')[0][1:])
            bboxes = dict_bbox[caseID]
            bbox_final = biggest_bbox(bboxes)
            size_r = bbox_final[1] - bbox_final[0]
            size_c = bbox_final[3] - bbox_final[2]
            args = 'convert ' + f + ' -crop ' + str(size_c) + 'x' + str(
                size_r) + '+' + str(bbox_final[2]) + '+' + str(bbox_final[0]) + ' ' + outname
            #print(args)
            os.system(args)

def crop_bbox_single(image, bbox, outname):
    size_r = bbox[1] - bbox[0]
    size_c = bbox[3] - bbox[2]
    args = 'convert ' + image + ' -crop ' + str(size_c) + 'x' + str(size_r) + '+' + str(bbox[2]) + '+' + str(bbox[0]) + ' ' + outname
    print(args)
    os.system(args)

def cross_valid_index(positives, negatives, n_splits=10):
    kf = KFold(n_splits=ns)
    train_pos, test_pos = kf.split(positives)
    train_neg, test_neg = kf.split(negatives)

    train_pos_set = [positives[x] for x in train_pos]
    test_pos_set = [positives[x] for x in test_pos]
    train_neg_set = [negatives[x] for x in train_neg]
    test_neg_set = [negatives[x] for x in test_neg]

    train = train_pos_set + train_neg_set
    train_L = [1] * len(train_pos_set) + [0] * (train_neg_set)

    test = test_pos_set + test_neg_set
    test_L = [1] * len(test_pos_set) + [0] * len(test_neg_set)
    return train, train_L, test, test_L


class ROI_Sampler:

    def __init__(self, roi_mat, caseID, window_size,
                 overlap, outdir, wsi_path=None,
                 roi_csv=None, wsi_size_csv=None,
                 dict_bbox=None, dict_wsi_size=None):
        self.roi_mat = roi_mat
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if dict_bbox is None:
            assert os.path.exists(roi_csv), "ROI csv file do not exist"
            self.dict_bbox = preprocess_roi_csv(roi_csv)
        if dict_wsi_size is None:
            assert os.path.exists(wsi_size_csv), "ROI size csv file do not exist"
            self.dict_wsi_size = preprocess_wsi_size_csv(wsi_size_csv)
        self.bboxes = self.dict_bbox[caseID]
        self.caseID = caseID
        self.wsi_size = self.dict_wsi_size[caseID]
        self.wsi_path = wsi_path
        self.window_size = window_size
        self.overlap = overlap
        self.bags = None
        self.pos_bags = None
        self.count = None
        self.pos_count = None
        self.neg_count = None
        self.outdir = outdir
        #self.negdir = os.path.join(self.outdir, 'neg')

    def sample_pos(self):
        assert os.path.exists(self.roi_mat), "ROI mat file do not exist"
        self.negdir = os.path.join(self.outdir, 'neg')
        if not os.path.exists(self.negdir): os.mkdir(self.negdir)
        self.posdir = os.path.join(self.outdir, 'pos')
        if not os.path.exists(self.posdir): os.mkdir(self.posdir)
        self.bags, self.pos_bags, self.count = self._sample_from_ROI_mat(self.roi_mat)
        self.pos_count, self.neg_count = self.count
        self.neg_count = self.pos_count - self.neg_count

        print("positive samples from " + str(self.caseID) + ' :' + str
            (self.pos_count))


    def sample_neg(self, neg_count=None, mode=None):
        mode_type = ['rand', 'relevant']
        assert mode in mode_type, "Enter valid mode type"
        bags = Bag(h=self.wsi_size[0], w=self.wsi_size[1],
                   size=self.window_size, overlap_pixel=self.overlap,
                   padded=False)

        if not neg_count:
            neg_count = self.neg_count
            assert self.neg_count is not None, "Need to run sample_pos first"
        if mode == 'rand':
            self.neg_bags = self._sample_negative_samples_rand(neg_count,
                                                                self.bboxes,
                                                                bags)
        else:
            pos_ind = self._bbox_to_bags_ind_in_wsi(self.bboxes, self.wsi_size,
                                                     self.window_size,
                                                     self.overlap)

            self.neg_bags = self._sample_negative_samples_relevant(neg_count, self.wsi_size, pos_ind, self.window_size, self.overlap)

            print(self.neg_bags)

            if self.wsi_path is not None:
                # need to crop roi and save in negdir
                for ind in self.neg_bags:
                    bbox = bags.bound_box(ind)
                    outname=str(self.caseID) + '_' + str(ind) + '.tif'
                    outname = os.path.join(self.negdir, outname)
                    crop_bbox_single(self.wsi_path, bbox, outname)


    def _sample_from_ROI_mat(self, mat_filename):
        im, M = load_mat(mat_filename)
        bags = Bag(im, padded=False)
        result = np.zeros(len(bags))
        pos_count = 0
        neg_count = 0
        for bag, i in bags:
            bbox = bags.bound_box(i)
            r, c, _ = bag.shape
            size = r * c
            #print(size)
            if np.sum(M[bbox[0]:bbox[1], bbox[2]:bbox[3]]) / size >= 0.7:
                result[i] = 1
                pos_count += 1
                cv2.imwrite(os.path.join(self.posdir, str(self.caseID) + '_' +
                   str(pos_count) + '.tif'), bag)
            else:
                if neg_count < pos_count:
                    neg_count += 1
                    cv2.imwrite(os.path.join(self.negdir, str(self.caseID) + '_'
                       + str(neg_count) + '.tif'), bag)

        return bags, result, [pos_count, neg_count]

    def _bbox_to_bags_ind_in_wsi(self, bboxes, WSI_size, window_size, overlap):
        """
            This function calculates the ROI index in terms of window(bag/word)
            sizes (i.e. given the size of WSI, give out a list with the index of
            bags that are contained in the ROI)

            Args:
                bboxes (List(n)): list of bounding boxes of a given image
                WSI_size [h (int), w (int)]: size of the WSI (height, width)
                window size (int): size of word/bags or any window of interest
                                  (usually
                                  3600 for bags and 120 for words)
                overlap (int): overlapping pixel in window

            Returns:
                result (List(m)): list with the index of bags that are contained in
                the ROI
        """
        # assumption is that we won't receive anything on the border where bags
        # can't fit
        h, w = WSI_size
        result = set()
        num_bag_w = int(math.floor((w - overlap) / (window_size - overlap)))
        num_bag_h = int(math.floor((h - overlap) / (window_size - overlap)))
        # Bounding box(int[]): [h_low, h_high, w_low, w_high]
        for h_low, h_high, w_low, w_high in bboxes:
            #if h_high >= h and w_high >= w:
            if h_high > h or w_high > w:
                print("Size incompatible for case: {}".format(self.caseID))
                print("Bounding box: {}, {}, {}, {}".format(h_low, h_high, w_low, w_high))
                print("WSI size: {}, {}". format(h, w))
                h_high = min(h, h_high)
                w_high = min(w, w_high)
            ind_w_low = int(max(math.floor((w_low - window_size) / (window_size
               - overlap) + 1), 0))
            ind_w_high = int(min(max(math.floor(w_high / (window_size -
                overlap)), 0), num_bag_w))
            ind_h_low = int(max(math.floor((h_low - window_size) / (window_size
               - overlap) + 1), 0))
            ind_h_high = int(min(max(math.floor(h_high / 
                (window_size - overlap)), 0), num_bag_h))
            for i in range(h_low, h_high+1):
                result.update(range(ind_h_low * num_bag_w + ind_w_low,
                   ind_h_high * num_bag_w + ind_w_high + 1))
        return np.sort(list(result))


    def _sample_negative_samples_relevant(self, num_of_neg_samples,
                                           WSI_size, roi_bags,
                                           window_size, overlap):
        """
            This function calculates sample negative samples next to the ROI

            Args:
                bboxes (List(n)): list of bounding boxes of a given image
                WSI_size [h (int), w (int)]: size of the WSI (height, width)
                roi_bags (List(m)): list with the index of roi bags that are
                contained
                                    in the ROI (Result from
                                    bbox_to_bags_ind_in_wsi)
                window size (int): size of word/bags or any window of interest
                                  (usually
                                  3600 for bags and 120 for words)
                overlap (int): overlapping pixel in window

            Returns:
                result (List(m)): List of index of sampled negative bags
        """
        assert roi_bags is not None and len(roi_bags) > 0, "invalid roi bags"
        h, w = WSI_size
        num_bag_w = int(math.floor((w - overlap) / (window_size - overlap)))
        length = math.floor(math.floor((h - overlap) / (window_size - overlap)) *
           math.floor((w - overlap) / (window_size - overlap)))
        count_left = num_of_neg_samples
        result = set()
        ind_list = list(range(length))

        i = 0

        while count_left > 0:
            # goes clockwise to sample bags
            ind = roi_bags[i]
            neigh = self._ROI_neighbor_not_roi(ind, num_bag_w, roi_bags,
                                                length)
            count_left -= len(neigh)
            result.update(neigh)
            roi_bags = np.concatenate([roi_bags, neigh])
            #roi_bags.extend(neigh)
            i += 1
            assert i < len(roi_bags), "ROI too big, not enough negative sample"
        print(result)
        return list(result)



    def _checkROI(self, idx, length, roi_bags):
        return idx >= 0 and idx < length and idx not in roi_bags

    def _ROI_neighbor_not_roi(self, idx, num_bag_w, roi_bags, length):
        result = []
        if self._checkROI(idx + 1, length, roi_bags):
            result += [int(idx + 1)]
        if self._checkROI(idx - 1, length, roi_bags):
            result += [int(idx - 1)]
        if self._checkROI(idx - num_bag_w, length, roi_bags):
            result += [int(idx - num_bag_w)]
        if self._checkROI(idx + num_bag_w, length, roi_bags):
            result += [int(idx + num_bag_w)]
        return result



    def _sample_negative_samples_rand(self, num_of_neg_samples, bboxes, bags):
        count_left = num_of_neg_samples
        ind_list = list(range(len(bags)))
        bbox = biggest_bbox(bboxes)
        result = np.zeros(num_of_neg_samples)
        while count_left > 0:
            i = np.random.choice(ind_list)
            ind_list.remove(i)
            num_intersected_pixel = 0

            # if bb[0] >= bbox[0] and bb[1] <= bbox[1] and bb[2] >= bbox[2] and bb[3] <= bbox[3]:
            # if row overlaps
            bb = bags.bound_box(i)
            roi_row = range(bbox[0], bbox[1])
            sample_row = set(range(bb[0], bb[1]))
            intersect_row = sample_row.intersection(roi_row)

            if len(intersect_row) > 0:
                # if col overlaps
                roi_col = range(bbox[2], bbox[3])
                sample_col = set(range(bb[2], bb[3]))
                intersect_col = sample_col.intersection(roi_col)

                num_intersected_pixel += len(intersect_col) * len(intersect_row)
                if num_intersected_pixel <= 0.2 * size:
                    result[num_of_neg_samples - count_left] = i
                    count_left -= 1
        return bags[result]
