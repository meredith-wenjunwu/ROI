import os
from util import *
import glob
from classifier import *

roi_path = '/projects/medical4/ximing/DistractorProject/page3/roi'
outdir = '/projects/medical4/ximing/DistractorProject/page3/out'
roi_csv='/projects/medical4/ximing/DistractorProject/csv/consensus.roi.alternative_new.csv'

window_size = 3600
overlap = 2400

wsi_size_csv = '/projects/medical4/ximing/DistractorProject/csv/wsi_size.csv'


roi_mats = glob.glob(os.path.join(roi_path, '*.mat'))
wsi_folder ='/projects/medical4/ximing/DistractorProject/page3/'

dict_bbox = preprocess_roi_csv(roi_csv)
dict_wsi_size = preprocess_wsi_size_csv(wsi_size_csv)

for roi_mat in roi_mats:
    bn = os.path.basename(roi_mat)
    caseID = int(os.path.splitext(bn)[0].split('_')[0])
    print(caseID)
    label = calculate_label_from_roi_bbox(dict_bbox[caseID], dict_wsi_size
        [caseID])

    save_path = os.path.join(wsi_folder, '{}_label.pkl'.format(caseID))
    pickle.dump(label, open(save_path, 'wb'))