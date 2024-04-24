from PIL import Image
import os
# from scipy.fftpack import dct, idct
import torch_dct as dct
import time
import numpy as np
import torch
# import torchvision.transforms as transforms
from torchvision.utils import save_image
import tqdm
from core.IntensPure_base_model import BaseModel
from torchvision.transforms import ToPILImage
from core.logger import LogTracker
import copy
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training() 

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        # self.cond_image = self.set_device(data.get('cond_image'))
        # print('cond_image', self.cond_image)
        self.no_normalization = self.set_device(data.get('no_normalization'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            # 'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            # 'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
            'gt_image': self.gt_image.detach()[:].float().cpu()*255,
            # 'cond_image': self.cond_image.detach()[:].float().cpu(),
        }
        # if self.task in ['inpainting','uncropping']:
        #     dict.update({
        #         'mask': self.mask.detach()[:].float().cpu(),
        #         'mask_image': (self.mask_image+1)/2,
        #     })
        if phase != 'train': 
            dict.update({
                # 'output': (self.output.detach()[:].float().cpu()+1)/2
                'output': self.output.detach()[:].float().cpu()*255
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        
        for train_data in tqdm.tqdm(self.phase_loader):
            
            self.set_input(train_data) 
                
            y_cond = self.gt_image.to('cuda')
            path = self.path
            
            self.optG.zero_grad()
            loss, y_noisy = self.netG(first_image = self.gt_image, y_cond = None)
            
            loss.backward()
            
            self.optG.step() 

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item()) 
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            # EMA : Exponential Moving Average
            if self.ema_scheduler is not None: 
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG) 

        for scheduler in self.schedulers:
            scheduler.step()
            
        return self.train_metrics.result(), y_noisy, y_cond, path
    
    def val_step(self, y_noisy, y_cond, path):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                self.path = path
                b, *_ = y_cond.shape
                
                patch_x0 = patchify_image(y_cond) # b,1024,c,8,8
                pdcx = dct.dct_2d(patch_x0)  #b,1024,c,8,8
                coeffs = []
                coeffs.append(pdcx[:,:,:,0,0].clone()) # DC
                coeffs.append(pdcx[:,:,:,1,1].clone()) # AC4
                
                coeffs.append(pdcx[:,:,:,0,1].clone()) # AC1
                coeffs.append(pdcx[:,:,:,0,2].clone()) # AC5
                
                coeffs.append(pdcx[:,:,:,1,0].clone()) # AC2
                coeffs.append(pdcx[:,:,:,2,0].clone()) # AC3
                
                channel_max=torch.zeros(len(coeffs), b) 
                channel_min=torch.zeros(len(coeffs), b)
                
                for i in range(len(coeffs)):
                    coeffs[i] = coeffs[i].reshape(b,512, 3, 1, 1)
                    for bb in range(b): 
                        channel_max[i,bb]=coeffs[i][bb].max()
                        channel_min[i,bb]=coeffs[i][bb].min()
                        coeffs[i][bb] = (coeffs[i][bb] - channel_min[i,bb]) / (channel_max[i,bb] - channel_min[i,bb])

                    coeffs[i] = depatchify_image(coeffs[i],(32,16),1) 
                    
                real_data = torch.cat(coeffs,dim=1) 
                real_data = real_data *2 -1
        
                assert -1 <= real_data.min() <= 0
                assert 0 <= real_data.max() <= 1
                
                if self.opt['distributed']: # X
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']: # X
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else: 
                        self.output, self.visuals = self.netG.restoration(y_cond=None, y_noisy=y_noisy, sample_num=self.sample_num, channel_max=channel_max, channel_min=channel_min)
                        real_base = patchify_image(self.output,1) #4,1024,18,1,1
                        real = (real_base + 1) / 2
                        for bb in range(b):
                            for j in range(int(real_base.size(2)/3)): 
                                g = j*3
                                real[bb,:,g:g+3] = real[bb,:,g:g+3] * (channel_max[j,bb] - channel_min[j,bb]) + channel_min[j,bb]

                        noise_reshape = real.reshape(b,512,18)
                        pdcx[:,:,:,0,0] =  noise_reshape[:,:,0:3]# DC RGB
                        pdcx[:,:,:,1,1] =  noise_reshape[:,:,3:6]# AC4 RGB
                        pdcx[:,:,:,0,1] =  noise_reshape[:,:,6:9]# AC1 RGB
                        pdcx[:,:,:,0,2] =  noise_reshape[:,:,9:12]# AC5 RGB
                        pdcx[:,:,:,1,0] =  noise_reshape[:,:,12:15]# AC2 RGB
                        pdcx[:,:,:,2,0] =  noise_reshape[:,:,15:18]# AC3 RGB
                        
                        noise_idct = dct.idct_2d(pdcx)

                        new_y_cond_cat_tensor = depatchify_image(noise_idct, (256, 128)).to('cuda') #32,64,3,32,32
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')
                
                cnt = 0
                test_save_dir = '/data/Palette_BDCT_Directional_toAC5_1/abc'
                
                for new_y_cond, cond in zip(new_y_cond_cat_tensor, y_cond) :
                    save_image(cond, os.path.join(test_save_dir, f'gt_{path[cnt]}'))
                    save_image(new_y_cond, os.path.join(test_save_dir, f'output_{path[cnt]}'))
                    cnt +=1
                
                self.gt_image = (y_cond - 0.5) / 0.5
                self.visuals = (self.visuals - 0.5) / 0.5
                self.output = (self.output - 0.5) / 0.5
                
                y_cond = (y_cond - 0.5) / 0.5
                new_y_cond_cat_tensor = (new_y_cond_cat_tensor - 0.5) / 0.5

                for met in self.metrics:
                    key = met.__name__
                    value = met(y_cond, new_y_cond_cat_tensor) 
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                    
                for key, value in self.get_current_visuals(phase='val').items(): 
                    self.writer.add_images(key, value)
                    
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()
    
    def test(self):
        
        
        self.netG.eval()
        self.test_metrics.reset()
        
        
        with torch.no_grad():
            for idx, phase_data in tqdm.tqdm(enumerate(self.phase_loader)):
                
                start = time.time()
                
                self.set_input(phase_data)
                b, *_ = self.gt_image.shape
                
                patch_x0 = patchify_image(self.gt_image) # b,1024,c,8,8
                pdcx = dct.dct_2d(patch_x0)  #b,1024,c,8,8
                coeffs = []
                coeffs.append(pdcx[:,:,:,0,0].clone()) # DC
                coeffs.append(pdcx[:,:,:,1,1].clone()) # AC4
                
                coeffs.append(pdcx[:,:,:,0,1].clone()) # AC1
                coeffs.append(pdcx[:,:,:,0,2].clone()) # AC5
                
                coeffs.append(pdcx[:,:,:,1,0].clone()) # AC2
                coeffs.append(pdcx[:,:,:,2,0].clone()) # AC3
                
                channel_max=torch.zeros(len(coeffs), b) 
                channel_min=torch.zeros(len(coeffs), b)
                
                for i in range(len(coeffs)):
                    coeffs[i] = coeffs[i].reshape(b,512, 3, 1, 1)
                    for bb in range(b): 
                        channel_max[i,bb]=coeffs[i][bb].max()
                        channel_min[i,bb]=coeffs[i][bb].min()
                        coeffs[i][bb] = (coeffs[i][bb] - channel_min[i,bb]) / (channel_max[i,bb] - channel_min[i,bb])

                    coeffs[i] = depatchify_image(coeffs[i],(32,16),1) 
                    
                real_data = torch.cat(coeffs,dim=1) 
                real_data = real_data *2 -1
                assert -1 <= real_data.min() <= 0
                assert 0 <= real_data.max() <= 1
                loss, y_noisy = self.netG(first_image = self.gt_image, y_cond = None, is_test=True)
                if self.opt['distributed']: 
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else: 
                    if self.task in ['inpainting','uncropping']: 
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else: 
                        self.output, self.visuals = self.netG.restoration(y_cond=None, y_noisy=y_noisy, sample_num=self.sample_num, is_test=True)

                        real_base = patchify_image(self.output,1) #4,1024,18,1,1
            
                        real = (real_base + 1) / 2
                        
                        for bb in range(b):
                            for j in range(int(real_base.size(2)/3)): 
                                g = j*3
                                real[bb,:,g:g+3] = real[bb,:,g:g+3] * (channel_max[j,bb] - channel_min[j,bb]) + channel_min[j,bb]

                        noise_reshape = real.reshape(b,512,18)
                        
                        real_result = torch.zeros(b,512,3,8,8).to('cuda')
                        
                        real_result[:,:,:,0,0] =  noise_reshape[:,:,0:3]# DC RGB
                        real_result[:,:,:,1,1] =  noise_reshape[:,:,3:6]# AC4 RGB
                        
                        real_result[:,:,:,0,1] =  noise_reshape[:,:,6:9]# AC1 RGB
                        real_result[:,:,:,0,2] =  noise_reshape[:,:,9:12]# AC5 RGB
                        
                        real_result[:,:,:,1,0] =  noise_reshape[:,:,12:15]# AC2 RGB
                        real_result[:,:,:,2,0] =  noise_reshape[:,:,15:18]# AC3 RGB
                        
                        noise_idct = dct.idct_2d(real_result)
                        new_y_cond_cat_tensor = depatchify_image(noise_idct, (256, 128)).to('cuda') #32,64,3,32,32

                        print(f'{time.time()-start:.4f} sec')
                        print()
                        batch_idx = 0
                        
                        for new_y_cond in new_y_cond_cat_tensor :
                            file_path = f'/data/Palette_BDCT_Directional_toAC5_1/duke_deep_eps12_ts36/{self.path[batch_idx]}'
                        
                            new_y_cond = new_y_cond.cpu() if new_y_cond.is_cuda else new_y_cond
                            save_image(new_y_cond, file_path, normarlize=False)
                            batch_idx += 1
                        
                self.iter += self.batch_size
                
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']: 
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__ 
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

def coefficients_separation_and_concatenation(bdct_result, stride=8) :
    dc_coefficient = bdct_result[::stride, ::stride, :]
    ac1_coefficient = bdct_result[::stride, 1::stride, :]
    ac2_coefficient = bdct_result[1::stride, ::stride, :]
    
    coefficients = []
    
    minmax = []

    dc_coefficient_normalized = ((dc_coefficient - np.min(dc_coefficient)) / (np.max(dc_coefficient) - np.min(dc_coefficient)) * 2) - 1
    # dc_coefficient_uint8 = dc_coefficient_normalized.astype(np.uint8)
    dc_coefficient_tensor = torch.tensor(dc_coefficient_normalized)

    ac1_coefficient_normalized = ((ac1_coefficient - np.min(ac1_coefficient)) / (np.max(ac1_coefficient) - np.min(ac1_coefficient)) * 2) - 1
    # ac1_coefficient_uint8 = ac1_coefficient_normalized.astype(np.uint8)
    ac1_coefficient_tensor = torch.tensor(ac1_coefficient_normalized)

    ac2_coefficient_normalized = ((ac2_coefficient - np.min(ac2_coefficient)) / (np.max(ac2_coefficient) - np.min(ac2_coefficient)) * 2) - 1
    # ac2_coefficient_uint8 = ac2_coefficient_normalized.astype(np.uint8)
    ac2_coefficient_tensor = torch.tensor(ac2_coefficient_normalized)
    
    for i in range(0, 3) :
        coefficients.append(dc_coefficient_tensor[:, :, i])
    for i in range(0, 3) :
        coefficients.append(ac1_coefficient_tensor[:, :, i])
    for i in range(0, 3) :
        coefficients.append(ac2_coefficient_tensor[:, :, i])
    
    concatenated = torch.stack(coefficients, dim=0)
    
    minmax.append(np.min(dc_coefficient))
    minmax.append(np.max(dc_coefficient))
    minmax.append(np.min(ac1_coefficient))
    minmax.append(np.max(ac1_coefficient))
    minmax.append(np.min(ac2_coefficient))
    minmax.append(np.max(ac2_coefficient))
    
    
    return dc_coefficient_tensor, ac1_coefficient_tensor, ac2_coefficient_tensor, concatenated, minmax

def minmax_inverse_scaling(coefficients, min_val, max_val):
    return (coefficients + 1) / 2 * (max_val - min_val) + min_val

def patchify_image(image, patch_size=8):
    # image: [batch_size, channels, height, width]
    batch_size, channels, height, width = image.size()

    # Calculate the number of patches in height and width
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Reshape the image to split it into patches on GPU
    image = image.view(batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size)

    # Permute dimensions to arrange patches properly
    image = image.permute(0, 2, 4, 1, 3, 5).contiguous()

    # Reshape to combine batch_size and num_patches_h * num_patches_w
    image = image.view(batch_size, num_patches_h * num_patches_w, channels, patch_size, patch_size)

    return image

def depatchify_image(patched_image, image_size=(256, 256), patch_size=8):
    # patched_image: [batch_size, num_patches, channels, patch_size, patch_size]
    batch_size, num_patches, channels, _, _ = patched_image.size()

    # Calculate the number of patches in height and width
    num_patches_h = image_size[0] // patch_size
    num_patches_w = image_size[1] // patch_size

    # Reshape the patched_image to combine patches
    patched_image = patched_image.view(batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)

    # Permute dimensions to arrange patches properly
    patched_image = patched_image.permute(0, 3, 1, 4, 2, 5).contiguous()

    # Reshape to combine batch_size and num_patches_h * num_patches_w
    patched_image = patched_image.view(batch_size, channels, image_size[0], image_size[1])

    return patched_image