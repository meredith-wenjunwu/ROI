#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:05:00 2019

@author: wuwenjun
"""
from argparse import ArgumentParser



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode', required=True,
                        default='kmeans',
                        const='kmeans',
                        nargs='?',
                        choices=['feature', 'kmeans', 'kmeans_visual',
                        'bag_of_words', 'classifier_train', 'classifier_test'],
                        help='Choose mode from k-means clustering, visualization and classification_training, classification_testing')
    parser.add_argument('--trained_kmeans_cluster', default=None,
       help='Previously trained kmeans clusters')
    parser.add_argument('--trained_hclusters', default=None,
        )
    parser.add_argument('--single_image', default=None, help='Input image path')
    parser.add_argument('--single_image_label', default='*.pkl', help="Label for training or testing for single image input")
    parser.add_argument('--image_folder', default=None, help='Input image batch folder' )
    parser.add_argument('--image_folder_label', default=None, help='Input image batch folder' )
    parser.add_argument('--image_format', required=True, default='.jpg', choices=['.jpg', '.png', '.tif', '.mat'],help='Input format')
    parser.add_argument('--save_intermediate', default=False, help='Whether or not to save the intermediate results')
    parser.add_argument('--dict_size', default= 40, help='Dictionary Size for KMeans')
    parser.add_argument('--histogram_bin', default=64, help='Bin size for histogram')
    parser.add_argument('--save_path', required=True, default='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test', help='save_path for outputs: e.g.features, kmeans, classifier')
    parser.add_argument('--word_size', default=120, help='Size of a word (in a bag of words model)')
    parser.add_argument('--bag_size', default=3600, help="Size of a bag (in a bag of words model)")
    parser.add_argument('--overlap_bag', default=2400, help='Overlapping pixels between bags')
    parser.add_argument('--ROI_csv', default=None, help='Input csv file for ROI tracking data')
    parser.add_argument('--WSI_csv', default=None, help='Input csv file for WSI size by case ID')
    parser.add_argument('--classifier', default='logistic',
                        const='logistic',
                        nargs='?',
                        choices=['logistic', 'svm'])
    parser.add_argument('--trained_model', default=None, help='previously trained model path')
    parser.add_argument('--lr', default=0.001, help='initial learning rate')
    parser.add_argument('--learning_rate', default='optimal', help='https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier')
    args = parser.parse_args()

    mode = args.mode
    image_path = args.single_image
    image_label_path = args.single_image_label
    folder_path = args.image_folder
    folder_label_path = args.image_label_path
    ext = args.image_format
    save_flag = args.save_intermediate
    dict_size = args.dict_size
    histogram_bin = args.histogram_bin
    save_path = args.save_path
    word_size = args.word_size
    roi_csv_file = args.ROI_csv
    wsi_csv_file = args.WSI_csv
    bag_size = args.bag_size
    overlap = args.overlap_bag
    clf_filename = args.trained_model
    kmeans = args.trained_cluster

    import cv2
    import numpy as np
    from feature import calculate_feature, get_histogram
    from cluster import *
    import os
    import pickle
    import glob
    from util import *
    from classifier import *

    if mode == 'kmeans':


        kmeans = pickle.load(open(kmeans, 'rb')) if kmeans is not None else None

        if image_path is not None and save_path is not None:
            # Save features
            filename = save_path + '_feat_word.pkl'
            result = get_feat_from_image(image_path, save_flag, word_size, fiename)

            # K-Means Part

            filename = save_path + '_kmeans.pkl'
            kmeans = construct_kmeans(result) if first_image else kmeans.partial_fit_kmeans(result, kmeans)

            pickle.dump(kmeans, open(filename, 'wb'))

        elif folder_path is not None and save_path is not None:
            print('-------Running Batch Job-------')
            # Feature computation and K-Means clustering in batch

            im_list = sorted(glob.glob(os.path.join(folder_path, '*' + ext)), key=os.path.getsize)
            count = 0
            print('# of images: %r' %(len(im_list)))
            # if the image processed is small, combine with the next image in queue
            small_image = False
            for im_p in im_list:
                if count % 5 == 0: print('Processed %r / %r' %(count, len(im_list)))
                count += 1
                # get filename without extension
                base = os.path.basename(im_p)
                path_noextend = os.path.splitext(base)[0]
                fname = path_noextend  + '_feat.pkl'
                filename = os.path.join(save_path, fname)
                if small_image: temp = result
                if os.path.exists(filename):
                    result = pickle.load(open(filename, 'rb'))
                else:
                    if ext != '.mat':
                        result = get_feat_from_image(im_p, save_flag, word_size, save_path=filename)
                    else:
                        im, m = load_mat(im_p)
                        result = get_feat_from_image(None, save_flag, word_size, image=im)
                        pickle.dump(result, open(filename, 'wb'))
                if small_image:
                    result = np.concatenate((result, temp), axis=0)
                    small_image = False
                if result.shape[0] < 200: small_image=True
                else: small_image = False
                # Online-Kmeans
                if kmeans is None and (not small_image):
                    print(result.shape)
                    kmeans = construct_kmeans(result)
                    assert kmeans is not None
                elif kmeans is not None and (not small_image):
                    kmeans = partial_fit_k_means(result, kmeans)
                    assert kmeans is not None, "kmeans construction/update invalid"

            hcluster = h_cluster(kmeans, final_size=dict_size)
            filename = os.path.join(save_path, 'kmeans.pkl')
            filename2 = os.path.join(save_path, 'hcluster.pkl')
            pickle.dump(kmeans, open(filename, 'wb'))
            pickle.dump(hcluster, open(filename2, 'wb'))

        else:
            print('Error: Input image path is None or save path is None.')
    elif mode == 'k-means-visualization':
        # Not implemented
        print('Not implmentd yet')

    elif mode == 'bag_of_words':
        assert folder_path is not None, "Need to provide path to images"
        kmeans_filename = os.path.join(save_path, 'kmeans.pkl')
        hcluster_filename = os.path.join(save_path, 'hcluster.pkl')
        assert os.path.exists(kmeans_filename) and os.path.exists(hcluster_filename), "Cannot find kmeans.pkl or hcluster.pkl"
        loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
        loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))
        im_list = glob.glob(os.path.join(folder_path, '*' + ext))
        
    elif mode == 'classifier-train':
        assert save_path is not None and os.path.exists(save_path), "Feature/kmeans path is None or does not exist"
        assert folder_path is not None or image_path is not None, "Error: Input image path is None or save path is None."
        assert loaded_kmeans is not None, "Path incorrect/File doesnt exist"

        if clf_filename is None:
            # initialize model
            start = True
            clf=model_init(args)
            clf_filename = os.path.join(save_path, 'clf.pkl')
        else:
            clf=model_load(clf_filename)
            start = False

        kmeans_filename = os.path.join(save_path, 'kmeans.pkl')
        hcluster_filename = os.path.join(save_path, 'hcluster.pkl')
        assert os.path.exists(kmeans_filename) and os.path.exists(hcluster_filename), "Cannot find kmeans.pkl or hcluster.pkl"
        loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
        loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))

        if folder_path is not None:
            pos_dir = os.path.join(sample_dir, 'pos')
            pos_files = glob.glob(os.path.join(pos_dir, '*.tif'))
            print('Number of positive samples: {}'.format(len(pos_files)))
            neg_dir = os.path.join(sample_dir, 'neg')
            neg_files = glob.glob(os.path.join(neg_dir, '*.tif'))
            print('Number of negative samples: {}'.format(len(neg_files)))
            while i < len(pos_files) or i < len(neg_files):
                if i % 10 == 0:
                    print("{} / {}".format(i, max(len(pos_files), len(neg_files))))
                if i < len(pos_files):
                    im_p = pos_files[i]
                    bag_feat = get_hist_from_image(im_p, loaded_kmeans,
                               loaded_hcluster, dict_size, word_size)
                    clf = model_update(clf, [bag_feat], [1], start)
                    if start:
                        start = False
                if i < len(neg_files):
                    im_p = neg_files[i]
                    bag_feat = get_hist_from_image(im_p, loaded_kmeans,
                               loaded_hcluster, dict_size, word_size)
                    clf = model_update(clf, [bag_feat], [0], start)
                    if start:
                        start = False
                i += 1
            model_save(clf, clf_filename)
        elif image_path is not None:
            assert image_label_path is not None and os.path.exists(image_label_path), "Error: invalid label file"

            print('Input training image: {}'.format(image_path))
            image = cv2.imread(image_path)
            image_label = pickle.load(open(image_label_path, 'rb'))
            assert image is not None, "imread fail, check path"
            image = np.array(image, dtype=int)
            bags = Bag(img=image, size=bag_size,
                       overlap_pixel=overlap, padded=True)
            assert len(bags) == len(image_label), "Label and input length does not match"
            for bag, i in bags:
                bag_feat = get_hist_from_image(None, loaded_kmeans,
                           loaded_hcluster, dict_size, word_size,
                           image=bag)
                clf = model_update(clf, [bag_feat], [label[i]], start=False)
                model_save(clf, clf_filename)

    elif mode == 'classifier-test':
        assert save_path is not None and os.path.exists(save_path), "Feature/kmeans path is None or does not exist"
        assert folder_path is not None or image_path is not None, "Error: Input image path is None or save path is None."
        assert loaded_kmeans is not None, "Path incorrect/File doesnt exist"
        assert csv_file is not None, "ROI tracking data not provided"
        assert clf_filename is not None, "Error: invalid input model"
        assert os.path.exists(clf_filename), "Error: invalid input model"

        clf=model_load(clf_filename)

        kmeans_filename = os.path.join(save_path, 'kmeans.pkl')
        hcluster_filename = os.path.join(save_path, 'hcluster.pkl')
        assert os.path.exists(kmeans_filename) and os.path.exists(hcluster_filename), "Cannot find kmeans.pkl or hcluster.pkl"
        loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
        loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))

        if folder_label_path is None:
            # Calculate label
            assert roi_csv_file is not None and os.path.exists(roi_csv_file), "ROI tracking data not provided"
            assert wsi_csv_file is not None and os.path.exists(wsi_csv_file), "ROI tracking data not provided"
            dict_bbox = preprocess_roi_csv(roi_csv)
            dict_wsi_size = preprocess_wsi_size_csv(wsi_size_csv)
            im_list = glob.glob(os.path.join(folder_path, '*' + ext))
            for im_p in im_list:
                base = os.path.basename(im_p)
                caseID = int(os.path.splitext(base)[0])
                print('-------Processing: {}-------'.format(caseID))
                label = calculate_label_from_roi_bbox(dict_bbox[caseID],                                  dict_wsi_size[caseID])
                label_path = os.path.join(folder_path, '{}_label.pkl'.format(caseID))
                pickle.dump(label_path, open(_path, 'wb'))
                print("Wrote label file to {}".format(label_path))
        else:
            im_list = glob.glob(os.path.join(folder_path, '*' + ext))
            metrics_list = [{'accuracy': 0, 'metrics':(0, 0, 0, 0)}]*len(im_list)
            index = 0
            for im_p in im_list:
                base = os.path.basename(im_p)
                caseID = int(os.path.splitext(base)[0])
                l_p = os.path.join(folder_label_path, '{}_label.pkl'.format(caseID))
                assert os.path.exists(l_p), "Did not find corresponding label file for {}".format(caseID)
                print('-------Processing: {}-------'.format(caseID))

                image = cv2.imread(im_p)
                image_label = pickle.load(open(l_p, 'rb'))
                assert image is not None, "imread fail, check path"
                image = np.array(image, dtype=int)
                bags = Bag(img=image, size=bag_size,
                           overlap_pixel=overlap, padded=True)
                assert len(bags) == len(image_label), "Label and input length does not match"
                result = np.zeros(len(bags))
                for bag, i in bags:
                    bag_feat = get_hist_from_image(None, loaded_kmeans,
                               loaded_hcluster, dict_size, word_size,
                               image=bag)
                    result[i] = model_predict(clf, [bag_feat])
                accuracy, metrics = model_report(result,
                                                 image_label, train=False)
                metrics_list[index]['accuracy'] = accuracy
                metrics_list[index]['metrics'] = metrics
                index += 1

            filename = save_path + '_test_result.pkl'
            pickle.dump(metrics_list, open(filename, 'wb'))









            # Feature computation and K-Means clustering in batch

            # im_list = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.tif"))
            # count = 0
            # for im_p in im_list:
            #     print('# of images: %r' %(len(im_list)))
            #     if count % 10 == 0: print('Processed %r / %r' %(count, len(im_list)))
            #     count += 1
            #     # get filename without extension
            #     base = os.path.basename(image_path)
            #     path_noextend = os.path.splitext(base)[0]
            #     caseID = int(path_noextend.split('_')[0][1:])
            #     print('CaseID: {}'.format(caseID))

            #     dict_bbox = preprocess_roi_csv(csv_file)
            #     assert dict_bbox.get(caseID) is not None, "case ID does not exist: check image name convention"
            #     feat_outpath = os.path.join(save_path, path_noextend + '_feat_bag.pkl')
            #     bag_feat, bags = get_hist_from_image(image_path, loaded_kmeans, dict_size, word_size, bag_size,
            #                                         overlap, save_flag, feat_outpath)



