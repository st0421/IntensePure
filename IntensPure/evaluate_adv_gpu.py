import scipy.io
import torch
import numpy as np
#import time
import os
from torchvision import datasets
import shutil
import argparse


#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index 
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--revised_query',type=str)
    opt = parser.parse_args()

    revised_query = opt.revised_query
    revised_query_f = 'query_f_'+revised_query
    result = scipy.io.loadmat('mat/'+revised_query+'.mat')
    data_dir = '../1.datasets/Market-1501/pytorch/'

    query_feature = torch.FloatTensor(result[revised_query_f])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in [revised_query]}


    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC

    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
