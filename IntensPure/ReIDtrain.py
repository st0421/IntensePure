# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
from model import ft_net
import yaml
from shutil import copyfile

version =  torch.__version__
#fp16

from pytorch_metric_learning import losses, miners #pip install pytorch-metric-learning

######################################################################
# Options
# --------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
    # data
    parser.add_argument('--data_dir',default='../1.datasets/Market-1501/pytorch',type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--DG', action='store_true', help='use extra DG-Market Dataset for training. Please download it from https://github.com/NVlabs/DG-Net#dg-market.' )
    # optimizer
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
    parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
    parser.add_argument('--cosine', action='store_true', help='use cosine lrRate' )
    parser.add_argument('--FSGD', action='store_true', help='use fused sgd, which will speed up trainig slightly. apex is needed.' )
    # backbone
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    # loss
    parser.add_argument('--warm_epoch', default=5, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--arcface', action='store_true', help='use ArcFace loss' )
    parser.add_argument('--circle', action='store_true', help='use Circle loss' )
    parser.add_argument('--cosface', action='store_true', help='use CosFace loss' )
    parser.add_argument('--contrast', action='store_true', help='use contrast loss' )
    parser.add_argument('--instance', action='store_true', help='use instance loss' )
    parser.add_argument('--ins_gamma', default=32, type=int, help='gamma for instance loss')
    parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
    parser.add_argument('--lifted', action='store_true', help='use lifted loss' )
    parser.add_argument('--sphere', action='store_true', help='use sphere loss' )
    parser.add_argument('--adv', default=0.0, type=float, help='use adv loss as 1.0' )
    parser.add_argument('--aiter', default=10, type=float, help='use adv loss with iter' )

    opt = parser.parse_args()

    data_dir = opt.data_dir
    name = opt.name
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)
    opt.gpu_ids = gpu_ids
    # set gpu ids
    if len(gpu_ids)>0:
        #torch.cuda.set_device(gpu_ids[0])
        cudnn.enabled = True
        cudnn.benchmark = True
    ######################################################################
    # Load Data
    # ---------
    #

    h, w = 256, 128

    transform_train_list = [
            transforms.Resize((h, w), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    transform_val_list = [
            transforms.Resize(size=(h, w),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]


    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose( transform_train_list ),
        'val': transforms.Compose(transform_val_list),
    }


    train_all = ''
    if opt.train_all:
        train_all = '_all'

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                            data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                            data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=True, num_workers=2, pin_memory=True,
                                                prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
                for x in ['train', 'val']}

    # Use extra DG-Market Dataset for training. Please download it from https://github.com/NVlabs/DG-Net#dg-market.
    if opt.DG:
        image_datasets['DG'] = DGFolder(os.path.join('../DG-Market' ),
                                            data_transforms['train'])
        dataloaders['DG'] = torch.utils.data.DataLoader(image_datasets['DG'], batch_size = max(8, opt.batchsize//2),
                                                shuffle=True, num_workers=2, pin_memory=True)
        DGloader_iter = enumerate(dataloaders['DG'])

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()

    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))
    print(time.time()-since)
    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.

    y_loss = {} # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        #best_model_wts = model.state_dict()
        #best_acc = 0.0
        warm_up = 0.1 # We start from the 0.1*lrRate
        warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
        if opt.arcface:
            criterion_arcface = losses.ArcFaceLoss(num_classes=opt.nclasses, embedding_size=512)
        if opt.cosface: 
            criterion_cosface = losses.CosFaceLoss(num_classes=opt.nclasses, embedding_size=512)
        if opt.triplet:
            miner = miners.MultiSimilarityMiner()
            criterion_triplet = losses.TripletMarginLoss(margin=0.3)
        if opt.lifted:
            criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
        if opt.contrast: 
            criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        if opt.sphere:
            criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=512, margin=4)
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                best_acc = 0
                # Iterate over data.
                for iter, data in enumerate(dataloaders[phase]):
                    # get the inputs
                    inputs, labels = data
                    now_batch_size,c,h,w = inputs.shape
                    if now_batch_size<opt.batchsize: # skip the last batch
                        continue
                    #print(inputs.shape)
                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    # if we use low precision, input also need to be fp16
                    #if fp16:
                    #    inputs = inputs.half()
    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == 'val':

                        start_ev=torch.cuda.Event(enable_timing=True)
                        end_ev=torch.cuda.Event(enable_timing=True)
                        with torch.no_grad():
                            start_ev.record()
                            outputs = model(inputs)
                            end_ev.record()

                        
                    else:
                        start_ev=torch.cuda.Event(enable_timing=True)
                        end_ev=torch.cuda.Event(enable_timing=True)
                        start_ev.record()
                        outputs = model(inputs)
                        end_ev.record()
                        torch.cuda.synchronize()
                        time_taken=start_ev.elapsed_time(end_ev)
                        print(time_taken*1e-3)
                        exit()

                    if opt.adv>0 and iter%opt.aiter==0: 
                        inputs_adv = ODFA(model, inputs)
                        outputs_adv = model(inputs_adv)

                    sm = nn.Softmax(dim=1)
                    log_sm = nn.LogSoftmax(dim=1)
                    return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere
                    if return_feature: 
                        logits, ff = outputs
                        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                        ff = ff.div(fnorm.expand_as(ff))
                        loss = criterion(logits, labels) 
                        _, preds = torch.max(logits.data, 1)
                        if opt.adv>0  and iter%opt.aiter==0:
                            logits_adv, _ = outputs_adv
                            loss += opt.adv * criterion(logits_adv, labels)
                        if opt.arcface:
                            loss +=  criterion_arcface(ff, labels)/now_batch_size
                        if opt.cosface:
                            loss +=  criterion_cosface(ff, labels)/now_batch_size
                        if opt.triplet:
                            hard_pairs = miner(ff, labels)
                            loss +=  criterion_triplet(ff, labels, hard_pairs) #/now_batch_size
                        if opt.lifted:
                            loss +=  criterion_lifted(ff, labels) #/now_batch_size
                        if opt.contrast:
                            loss +=  criterion_contrast(ff, labels) #/now_batch_size
                        if opt.sphere:
                            loss +=  criterion_sphere(ff, labels)/now_batch_size
                    elif opt.PCB:  #  PCB
                        part = {}
                        num_part = 6
                        for i in range(num_part):
                            part[i] = outputs[i]

                        score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                        _, preds = torch.max(score.data, 1)

                        loss = criterion(part[0], labels)
                        for i in range(num_part-1):
                            loss += criterion(part[i+1], labels)
                    else:  #  norm
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        if opt.adv>0 and iter%opt.aiter==0:
                            loss += opt.adv * criterion(outputs_adv, labels)

                    del inputs
                    # use extra DG Dataset (https://github.com/NVlabs/DG-Net#dg-market)
                    if opt.DG and phase == 'train' and epoch > num_epochs*0.1:
                        try:
                            _, batch = DGloader_iter.__next__()
                        except StopIteration: 
                            DGloader_iter = enumerate(dataloaders['DG'])
                            _, batch = DGloader_iter.__next__()
                        except UnboundLocalError:  # first iteration
                            DGloader_iter = enumerate(dataloaders['DG'])
                            _, batch = DGloader_iter.__next__()
                            
                        inputs1, inputs2, _ = batch
                        inputs1 = inputs1.cuda().detach()
                        inputs2 = inputs2.cuda().detach()
                        # use memory in vivo loss (https://arxiv.org/abs/1912.11164)
                        outputs1 = model(inputs1)
                        if return_feature:
                            outputs1, _ = outputs1
                        elif opt.PCB:
                            for i in range(num_part):
                                part[i] = outputs1[i]
                            outputs1 = part[0] + part[1] + part[2] + part[3] + part[4] + part[5]
                        outputs2 = model(inputs2)
                        if return_feature:
                            outputs2, _ = outputs2
                        elif opt.PCB:
                            for i in range(num_part):
                                part[i] = outputs2[i]
                            outputs2 = part[0] + part[1] + part[2] + part[3] + part[4] + part[5]

                        mean_pred = sm(outputs1 + outputs2)
                        kl_loss = nn.KLDivLoss(size_average=False)
                        reg= (kl_loss(log_sm(outputs2) , mean_pred)  + kl_loss(log_sm(outputs1) , mean_pred))/2
                        loss += 0.01*reg
                        del inputs1, inputs2
                        #print(0.01*reg)
                    # backward + optimize only if in training phase
                    if epoch<opt.warm_epoch and phase == 'train': 
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss = loss*warm_up

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                    if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += loss.item() * now_batch_size
                    else :  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size
                    del loss
                    running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0-epoch_acc)            
                # deep copy the model
#                if phase == 'val' and epoch%10 == 9:
                if phase == 'val' and best_acc < epoch_acc:
                    best_acc = epoch_acc
                    last_model_wts = model.state_dict()
                    if len(opt.gpu_ids)>1:
                        save_network(model.module, epoch+1)
                    else:
                        save_network(model, epoch+1)
                if phase == 'val':
                    draw_curve(epoch)
                if phase == 'train':
                    scheduler.step()
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(last_model_wts)
        if len(opt.gpu_ids)>1:
            save_network(model.module, 'last')
        else:
            save_network(model, 'last')

        return model


    ######################################################################
    # Draw Curve
    #---------------------------
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig( os.path.join('./model',name,'train.jpg'))

    ######################################################################
    # Save model
    #---------------------------
    def save_network(network, epoch_label):
        save_filename = 'net_%s.pth'% epoch_label
        save_path = os.path.join('./model',name,save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda(gpu_ids[0])


    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #

    return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere
    model = ft_net(len(class_names), opt.droprate, opt.stride, circle = return_feature, ibn=opt.ibn, linear_num=opt.linear_num)
        
    opt.nclasses = len(class_names)

    print(model)

    # model to gpu
    model = model.cuda()

    optim_name = optim.SGD #apex.optimizers.FusedSGD
    if opt.FSGD: # apex is needed
        optim_name = FusedSGD

    if len(opt.gpu_ids)>1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
        if not opt.PCB:
            ignored_params = list(map(id, model.module.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
            classifier_params = model.module.classifier.parameters()
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, model.module.model.fc.parameters() ))
            ignored_params += (list(map(id, model.module.classifier0.parameters() ))
                        +list(map(id, model.module.classifier1.parameters() ))
                        +list(map(id, model.module.classifier2.parameters() ))
                        +list(map(id, model.module.classifier3.parameters() ))
                        +list(map(id, model.module.classifier4.parameters() ))
                        +list(map(id, model.module.classifier5.parameters() ))
                        #+list(map(id, model.module.classifier6.parameters() ))
                        #+list(map(id, model.module.classifier7.parameters() ))
                        )
            base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
            classifier_params = filter(lambda p: id(p) in ignored_params, model.module.parameters())
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    else:
        if not opt.PCB:
            ignored_params = list(map(id, model.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            classifier_params = model.classifier.parameters()
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, model.model.fc.parameters() ))
            ignored_params += (list(map(id, model.classifier0.parameters() )) 
                        +list(map(id, model.classifier1.parameters() ))
                        +list(map(id, model.classifier2.parameters() ))
                        +list(map(id, model.classifier3.parameters() ))
                        +list(map(id, model.classifier4.parameters() ))
                        +list(map(id, model.classifier5.parameters() ))
                        #+list(map(id, model.classifier6.parameters() ))
                        #+list(map(id, model.classifier7.parameters() ))
                        )
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            classifier_params = filter(lambda p: id(p) in ignored_params, model.parameters())
            optimizer_ft = optim_name([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': classifier_params, 'lr': opt.lr}
            ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)
    if opt.cosine:
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, opt.total_epoch, eta_min=0.01*opt.lr)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 1-2 hours on GPU. 
    #
    dir_name = os.path.join('./model',name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    #record every run
    copyfile('./train.py', dir_name+'/train.py')
    copyfile('./model.py', dir_name+'/model.py')

    # save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    criterion = nn.CrossEntropyLoss()
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=opt.total_epoch)

