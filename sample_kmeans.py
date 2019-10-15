import os
from util import *
import glob
from classifier import *
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument('--trained_model', default=None, help='previously trained model path')
parser.add_argument('--lr', default=0.001, help='initial learning rate')
parser.add_argument('--learning_rate', default='optimal', help='https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier')
parser.add_argument('--classifier', default='logistic',
                    const='logistic',
                    nargs='?',
                    choices=['logistic', 'svm'])
args = parser.parse_args()

save_path = '/projects/medical4/ximing/DistractorProject/feature_page3'
kmeans_filename = os.path.join(save_path, 'kmeans.pkl')
hcluster_filename = os.path.join(save_path, 'hcluster.pkl')
clf_filename = os.path.join(save_path, 'clf.pkl')
dict_size = 40
histogram_bin = 64
word_size = 120
bag_size = 3600
overlap = 2400
save_flag = False

loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))

sample_dir = '/projects/medical4/ximing/DistractorProject/page3/out'
processed = []
pos_dir = os.path.join(sample_dir, 'pos')
pos_files = glob.glob(os.path.join(pos_dir, '*.tif'))
print('Number of positive samples: {}'.format(len(pos_files)))
neg_dir = os.path.join(sample_dir, 'neg')
neg_files = glob.glob(os.path.join(neg_dir, '*.tif'))
print('Number of negative samples: {}'.format(len(neg_files)))



clf = model_init(args)
start = True

# clf = model_load(clf_filename)

i = 0

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
