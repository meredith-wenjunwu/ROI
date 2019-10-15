import glob
import os
from util import *


roi_path = '/projects/medical4/ximing/DistractorProject/page3/roi'  
outdir = '/projects/medical4/ximing/DistractorProject/page3/out'
roi_csv='/projects/medical4/ximing/DistractorProject/csv/consensus.roi.alternative_new.csv'
window_size=3600
overlap=2400
roi_size_csv = '/projects/medical4/ximing/DistractorProject/csv/wsi_size.csv'

processed = []

roi_mats = glob.glob(os.path.join(roi_path, '*.mat'))
wsi_folder ='/projects/medical4/ximing/DistractorProject/page3/'

for roi_mat in roi_mats:
    bn = os.path.basename(roi_mat)
    caseID = int(os.path.splitext(bn)[0].split('_')[0])
    if caseID not in processed:
        fn = '*' + str(caseID) + '_*.tif'
        wsi_path = glob.glob(os.path.join(wsi_folder, fn))
        print(caseID)
        if len(wsi_path) == 0:
            print(os.path.join(wsi_folder, fn))
        assert len(wsi_path) > 0, "Cannot find corresponding WSI"
        wsi_path = wsi_path[0]
        rs = ROI_Sampler(roi_mat, caseID, window_size, overlap, outdir, wsi_path,
           roi_csv, roi_size_csv)
        rs.sample_pos()
        rs.sample_neg(mode='relevant')
