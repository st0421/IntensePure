import argparse
import scipy.io
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '2'

from PIL import Image
from model import ft_net, Backbone_nFC
from torchvision import datasets, transforms

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--name', default='resnet50', type=str, help='save model path')
parser.add_argument('--ar_classes',default=30, type=int)
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')

parser.add_argument('--test_dir',default='../1.datasets/Market-1501',type=str, help='./test_data')

opts = parser.parse_args()
name = opts.name

data_dir = opts.test_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################################################
# sort the images
def scoring(qf,gf):
    query = qf.view(-1,1)
    gf = gf.view(1,-1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score    

def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index, score[index]

def sort_img_inv(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)  #from small to large
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)   
    junk_index2 = np.intersect1d(query_index, camera_index) 
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index, score[index]
    
########################################################################

data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_network(ARnetwork):
    AR_save_path = os.path.join('./model',name,'AR_net_%s.pth'%opts.which_epoch)
    ARnetwork.load_state_dict(torch.load(AR_save_path))

    return ARnetwork


AR_model_structure = Backbone_nFC(opts.ar_classes, name)
ARmodel = load_network(AR_model_structure)
# Change to test mode
ARmodel = ARmodel.eval()
if torch.cuda.is_available():
    ARmodel = ARmodel.to(device)
########################################################################


file_name = ['query','deep_mis_ranking_eps2','deep_mis_ranking_eps4','deep_mis_ranking_eps8','deep_mis_ranking_eps12','deep_mis_ranking_eps16']

with open("data/deep_mis.txt",'a') as f:

    for enum, fname in enumerate(file_name):
        label = [0,2,4,8,12,16]
        print("start process: ",label[enum])

        result = scipy.io.loadmat('mat/'+fname+'.mat')
        image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,'pytorch',x) ) for x in ['gallery',fname]}
        if fname == 'query':
            query_feature = torch.FloatTensor(result['query_f'])
        else:
            query_feature = torch.FloatTensor(result['query_f_'+fname])
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]

        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]

        query_feature = query_feature.to(device)
        gallery_feature = gallery_feature.to(device)


        Q_R_score = []
        AR_err = []
        imgs=[]
        temp = ""
        AR_consistency =[]
        ID_consistency = []
        for i in range(len(query_label)):
            query_path, _ = image_datasets[fname].imgs[i]
            index, score = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)            
            input_img = Image.open(query_path)
            input_img = data_transforms(input_img).unsqueeze(0).to(device)
            imgs.append(input_img.clone())

        ##rank1 process
            for q_idx in range(10):
                rank_path, _ = image_datasets['gallery'].imgs[index[q_idx]] 
                rimg = Image.open(rank_path)
                rimg = data_transforms(rimg).unsqueeze(0).to(device)
                imgs.append(rimg.clone())
            for q_idx in range(10):
                rank_path, _ = image_datasets['gallery'].imgs[index[::-1][q_idx]]
                rimg = Image.open(rank_path)
                rimg = data_transforms(rimg).unsqueeze(0).to(device)
                imgs.append(rimg.clone()) 

            Q_R_score = [round(score[r], 4) for r in range(10)]
            
            for q_idx in range(10):
                for a in range(q_idx+1, 10):
                    score_between_query_ranked = scoring(gallery_feature[index[q_idx]],gallery_feature[index[a]])
                    Q_R_score.append(round(score_between_query_ranked[0],4))

            for q_idx in range(10):
                for a in range(q_idx+1, 10):
                    score_between_query_ranked = scoring(gallery_feature[index[::-1][q_idx]],gallery_feature[index[a]])
                    Q_R_score.append(round(score_between_query_ranked[0],4))

        ##총 이미지 11장 저장 query+top10
            input_imgs = torch.cat(imgs,dim=0)
            outputs = ARmodel(input_imgs)   #[31,30]
            outputs = outputs.cpu().detach().numpy()
            outputs = np.round(outputs, decimals=4)
            Q_R1_ARerr = [0.0 if abs(result) < 0.0001 else round(abs(result), 4) for result in (outputs[0] - outputs[1])]
            Q_Ravg_err = [0.0 if abs(result) < 0.0001 else round(abs(result), 4) for result in (outputs[0] - np.mean(outputs[1:21], axis=0))]
            inv_ARerr = [0.0 if abs(result) < 0.0001 else round(abs(result), 4) for result in (outputs[0] - np.mean(outputs[21:], axis=0))]

            AR_err.append(Q_R1_ARerr)
            AR_err.append(Q_Ravg_err)
            AR_err.append(inv_ARerr)

            for t in Q_R_score:
                temp += str(t)+" "
            temp+='\t'
            for step in range(len(AR_err)):
                for e in AR_err[step]:
                    temp+= str(e)+" "
            temp+='\t'
            temp+= str(label[enum])+'\n'
            Q_R_score.clear()
            AR_err.clear()
            imgs.clear()
        f.write(temp)
        print("end process: ",label[enum])



