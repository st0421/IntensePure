import os
import warnings
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import init


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='dataset path.')
    parser.add_argument('--data', type=str, default='appen_deep_mis.txt')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--input_dim', type=int, default=190)

    return parser.parse_args()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ADDataSet(Dataset):
    def __init__(self, path='data', file=''):
        self.path = path
        self.file = file

        df = pd.read_csv(os.path.join(self.path, self.file), sep='\t', header=None,names=['ID_con','AR_err','label'])
        
        selected_columns = df[['ID_con', 'AR_err']]
        
        self.data = selected_columns.apply(lambda row: [float(value) for value in row.values[0].split()] + [float(value) for value in row.values[1].split()], axis=1).tolist()
       
        self.label = df[['label']].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        print(f' distribution of samples: {self.sample_counts}')

    def __len__(self):
        return sum(self.sample_counts)
    
    def get_labels(self):
        return self.label
    
    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.tensor(data, dtype=torch.float32)
        label = self.label[idx].astype(np.int32)

        return data, label

class AAIE(nn.Module): #Adversarial Attack Intensity Estimator

    def __init__(self,input_dim=190,class_num=1,num_bottleneck=256):
        super(AAIE,self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, num_bottleneck*2).apply(weights_init_kaiming)
        self.bn1 = nn.BatchNorm1d(num_bottleneck*2).apply(weights_init_kaiming)
        self.relu = nn.LeakyReLU(0.1).apply(weights_init_kaiming)
        self.fc3 = nn.Linear(num_bottleneck*2, num_bottleneck).apply(weights_init_kaiming)
        self.bn3 = nn.BatchNorm1d(num_bottleneck).apply(weights_init_kaiming)
        self.relu3 = nn.LeakyReLU(0.1).apply(weights_init_kaiming)
        
        self.fc2 = nn.Linear(num_bottleneck, class_num).apply(weights_init_classifier)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)        
  
        x = self.fc2(x)
        return x
    
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    model = AAIE(input_dim=args.input_dim).to(device)

    dataset = ADDataSet(args.path, args.data)    
    dataset_length = len(dataset)
    train_size = int(len(dataset)*0.8)
    val_size = dataset_length - train_size
    train_dataset , val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, 
                                batch_size = args.batch_size,
                                num_workers = args.workers,
                                shuffle = True,  
                                pin_memory = True)

    val_loader = DataLoader(val_dataset,
                            batch_size = args.batch_size,
                            num_workers = args.workers,
                            shuffle = False,  
                            pin_memory = True)
    
    #criterion_mse = torch.nn.MSELoss().to(device)
    criterion_mse = torch.nn.L1Loss().to(device)
    criterion_mae = torch.nn.L1Loss().to(device)
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


    loss_meter = AverageMeter()
    error_meter = AverageMeter()

    best_err = 100
    for epoch in tqdm(range(1, args.epochs + 1)):
        iter_cnt = 0
        correct_sum = 0
        all_targets = []
        all_predicts = []
            
        model.train()
        for (data, targets) in train_loader:

            iter_cnt += 1
            optimizer.zero_grad()


            data = data.to(device)
            targets = targets.to(device)
            out = model(data)
            loss = criterion_mse(out,targets)
            loss.backward()
            optimizer.step()

            error = criterion_mae(out,targets)

            num = data.size(0)
            loss_meter.update(loss.item(), num)
            error_meter.update(error.item(), num)

            predicts = torch.abs(torch.round(out)).to(torch.int)
            all_targets.append(targets[0].cpu().detach().numpy())
            all_predicts.append(predicts[0].cpu().detach().numpy())
             

            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            acc = correct_sum.float() / float(train_dataset.__len__())
       
        tqdm.write('Train: [Epoch %d] Error: %.4f. Loss: %.3f. Acc: %.3f' % (epoch, error_meter.val, loss_meter.val, acc))
        
        with torch.no_grad():
            iter_cnt = 0
            correct_sum = 0
            all_targets = []
            all_predicts = []
            
            model.eval()
            start_ev=torch.cuda.Event(enable_timing=True)
            end_ev=torch.cuda.Event(enable_timing=True)
            for (data, targets) in val_loader:
                data = data.to(device)
                targets = targets.to(device)
                start_ev.record()
                out = model(data)
                end_ev.record()
                loss = criterion_mse(out,targets)
                iter_cnt+=1
                
                error = criterion_mae(out,targets)
                num = data.size(0)
                loss_meter.update(loss.item(), num)
                error_meter.update(error.item(), num)
                predicts = torch.abs(torch.round(out)).to(torch.int)
                all_targets.append(targets[0].cpu().detach().numpy())
                all_predicts.append(predicts[0].cpu().detach().numpy())
             
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num
            acc = correct_sum.float() / float(val_dataset.__len__())

            scheduler.step()
                

            best_err = min(error_meter.val,best_err)

            tqdm.write('Val: [Epoch %d] Error: %.4f. Loss: %.3f. Acc: %.3f' % (epoch, error_meter.val, loss_meter.val, acc))
            
            if error_meter.val == best_err:
                if error_meter.val < 1:
                    torch.save({'iter': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                os.path.join('estimator_checkpoints', "err"+str(error_meter.val)+"_epoch"+str(epoch)+"_"+args.data[:-4]+".pth"))
                    tqdm.write('Model saved.')
        
if __name__ == "__main__":        
    run_training()