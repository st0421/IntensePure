from PIL import Image
# from scipy.fftpack import dct, idct
# import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch_dct as dct
import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet_dc, unet_ac1, unet_ac2, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn_dc = UNet(**unet_dc)
        self.denoise_fn_ac1 = UNet(**unet_ac1)
        self.denoise_fn_ac2 = UNet(**unet_ac2)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas 
        timesteps, = betas.shape 
        self.num_timesteps = int(timesteps)
        gammas = np.cumprod(alphas, axis=0) 
        gammas_prev = np.append(1., gammas[:-1]) 
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t - extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise)

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat + extract(self.posterior_mean_coef2, t, y_t.shape) * y_t)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None, unetflag = None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        denoise_fn = None
        if unetflag == 'dc' :
            denoise_fn = self.denoise_fn_dc
        elif unetflag == 'ac1' :
            denoise_fn = self.denoise_fn_ac1
        elif unetflag == 'ac2' :
            denoise_fn = self.denoise_fn_ac2
        # y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))
        y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=denoise_fn(y_t, noise_level))
        
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(y_0_hat=y_0_hat, y_t=y_t, t=t)
        
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        # print('sample_gammas :', sample_gammas.shape)
        return (sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise)

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None, unetflag = None):
        model_mean, model_log_variance = self.p_mean_variance(y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, unetflag=unetflag)
        
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()



    @torch.no_grad()
    def restoration(self, y_cond, y_noisy=None, y_0=None, mask=None, sample_num=8, channel_max=None, channel_min=None, is_test=False):
        b, *_ = y_noisy.shape
        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        if is_test == False : 
            real_base = patchify_image(y_noisy,1) 
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
            gt = depatchify_image(noise_idct, (256, 128)).to('cuda') #32,64,3,32,32
            ret_arr = gt 
        else : 
            y_noisy = y_noisy.to('cuda')
            ret_arr = y_noisy
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device='cuda', dtype=torch.long)
            noise_dc = y_noisy[:, 0:6, :, :]
            noise_ac1 = y_noisy[:, 6:12, :, :]
            noise_ac2 = y_noisy[:, 12:18, :, :]
            noise_dc = self.p_sample(noise_dc, t, y_cond=None, unetflag='dc') 
            noise_ac1 = self.p_sample(noise_ac1, t, y_cond=None, unetflag='ac1') 
            noise_ac2 = self.p_sample(noise_ac2, t, y_cond=None, unetflag='ac2') 
            y_noisy[:, 0:6, :, :] = noise_dc
            y_noisy[:, 6:12, :, :] = noise_ac1
            y_noisy[:, 12:18, :, :] = noise_ac2
            if is_test == False :
                if i % sample_inter == 0:
                    real_base = patchify_image(y_noisy,1) #4,512,18,1,1
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
                    gt = depatchify_image(noise_idct, (256, 128)).to('cuda') # 32,64,3,32,32
                    ret_arr = torch.cat([ret_arr, gt], dim=0)
        return y_noisy, ret_arr
    
    def forward(self, first_image, y_cond=None, mask=None, noise=None, is_test=False):
        # sampling from p(gammas)
        b, *_ = first_image.shape
        # print(first_image.shape) # B, C, H, W
        if is_test==False :
            t = torch.randint(1, self.num_timesteps, (b,), device=first_image.device).long() 
        else : 
            t = (torch.ones((b,), device=first_image.device)*self.num_timesteps-1).long() 
        # self.gammas 
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1)) 
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=first_image.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)
        patch_x0 = patchify_image(first_image) # b,512,c,8,8
        pdcx = dct.dct_2d(patch_x0)  #b,512,c,8,8 
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
                # print(channel_max.shape)
            coeffs[i] = depatchify_image(coeffs[i],(32,16),1) 
        real_data = torch.cat(coeffs,dim=1) 
        real_data = real_data *2 -1
        noise = default(noise, lambda: torch.randn_like(real_data)) 
        y_noisy = self.q_sample(y_0=real_data, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise) 
        noise_plain = y_noisy[:, 0:6, :, :]
        noise_vertical = y_noisy[:, 6:12, :, :]
        noise_horizontal = y_noisy[:, 12:18, :, :]
        noise_plain = self.denoise_fn_dc(noise_plain, sample_gammas) 
        noise_vertical = self.denoise_fn_ac1(noise_vertical, sample_gammas) 
        noise_horizontal = self.denoise_fn_ac2(noise_horizontal, sample_gammas) 
        noise_hat = torch.cat([noise_plain, noise_vertical, noise_horizontal], dim=1)
        loss = self.loss_fn(noise, noise_hat)
        return loss, y_noisy 
    
# gaussian diffusion trainer class
def exists(x):#
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t) 
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) 

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

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