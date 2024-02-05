# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:09:41 2021

@author: Kevin Tsia
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 19:23:48 2021

@author: Kevin Tsia
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:56:29 2021

@author: Kevin Tsia
"""
import random
import numpy as np
import argparse
import os
from os import path
from tqdm import tqdm
import time
import copy
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import torchvision
from sklearn.preprocessing import StandardScaler
from skimage import color

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import mean_squared_error
from gan_training import utils
from gan_training.utils import return_data_test

from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints_test import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval_test import DisentEvaluator, Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler
)
#from dvae import dvae

# Arguments

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('--config_dir', default='./configs', type=str, help='Path to configs directory')
parser.add_argument('--output_dir', default='Z:/COVID-FTP/Rashmi/ID-GANCheckpoints/outputs/', type=str, help='Path to outputs directory')
parser.add_argument('--dvae_name', default='none', type=str, help='Name of the experiment for pre-training disentangling VAE')
parser.add_argument('--config', type=str, help='Name of base config file')
parser.add_argument('--name', type=str, help='Name of the experiment')
parser.add_argument('--nf', '--nfilter', default=-1, type=int, help='Base number of filters')
parser.add_argument('--batch_size','--bs', default=10, type=int, help='Batch size')
parser.add_argument('--nc', default=3, type=int, help='Batch size')

parser.add_argument('--reg_param', default=-1, type=float, help='R1 regularization parameter')
parser.add_argument('--w_info', default=-1, type=float, help='weighting constant on ID Loss')
parser.add_argument('--mi', '--max_iter', default=-1, type=int, help='Max training iteration')
parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
#parser.add_argument('--batch_size', default=64, type=int, help='batch size')

parser.add_argument('--image_size', default=256, type=int, help='Batch size') ##This is the image size for which the autoencoder is designed to

parser.add_argument('--dset_dir', default='C:/Users/Rashmi/ProcessedData/', type=str, help='dataset directory')
parser.add_argument('--dataset', default='LiveCell/Selected/', type=str, help='dataset name') #CellCycle/Vero/LC
parser.add_argument('--datatype', default='rgb', type=str, help='dataset name')
#parser.add_argument('--img_size', default=256, type=int, help='Image synthesis size')

parser.add_argument('--c_dim', default=10, type=int, help='Image synthesis size')
parser.add_argument('--z_dim', default=256, type=int, help='Image synthesis size')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda')
parser.add_argument('--infodistil_mode', default=True, help='Infodistill')

#parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda')
parser.add_argument('--seed', default=1, type=int, help='Random Seed')

args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#print(args.config)
config_path = os.path.join(args.config_dir, args.config)
config = load_config(config_path)
#is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# = = = = = Customized Configurations = = = = = #
out_dir = os.path.join(args.output_dir, args.name)
config['training']['out_dir'] = out_dir
if args.nf > 0:
    config['generator']['kwargs']['nfilter'] = args.nf 
    config['discriminator']['kwargs']['nfilter'] = args.nf 
if args.batch_size > 0:
    config['training']['batch_size'] = args.batch_size
if args.reg_param > 0:
    config['training']['reg_param'] = args.reg_param
if args.w_info > 0:
    config['training']['w_info'] = args.w_info
if args.mi > 0:
    max_iter = config['training']['max_iter'] = args.mi
else:
    max_iter = config['training']['max_iter']
if args.dvae_name != 'none':
    config['dvae']['runname'] = args.dvae_name
# = = = = = Customized Configurations = = = = = #

# Short hands
batch_size = 100
d_steps = config['training']['d_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']
sample_size = config['test']['sample_size']
out_dir = config['training']['out_dir']
#print('out_dir=',out_dir)
checkpoint_dir = path.join(out_dir, 'chkpts')
#print('checkpoint_dir',checkpoint_dir)
# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

#device = torch.device("cuda:0" if is_cuda else "cpu")
device= torch.device('cpu')
'''train_dataset = get_dataset(
    name=config['data']['type'],
    data_dir='Z:/COVID-FTP/Rashmi/data/Test_Reconstruction/LCGray/',
    size=args.img_size,
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        num_workers=0,
        shuffle=False, pin_memory=True, sampler=None, drop_last=True
)'''
data_loader = return_data_test(args)

# Create models
dvae, fvae, generator, discriminator = build_models(config, args)


selectVAE='fvae'

if selectVAE=='dvae':
    dvae_ckpt_path = os.path.join(args.output_dir, args.dvae_name, 'chkpts', config['dvae']['chkptname'])

    dvae_ckpt = torch.load(dvae_ckpt_path)['model_states']['net']
    #for key in list(dvae_ckpt.keys()):
    #    dvae_ckpt[key.replace('decoder.', 'module.decoder.'). replace('encoder.', 'module.encoder.')] = dvae_ckpt.pop(key)

    dvae.load_state_dict(dvae_ckpt)
else:

    fvae_ckpt_path = os.path.join(args.output_dir, args.dvae_name,'chkpts', config['fvae']['chkptname'])

    fvae_ckpt = torch.load(fvae_ckpt_path, map_location=torch.device('cpu'))

    for key in list(fvae_ckpt['model_states']['VAE'].keys()):
        fvae_ckpt['model_states']['VAE'][key.replace('decode.', 'decoder.layer.').
                                         replace('decode.', 'decoder.layer.').
                                         replace('decode.', 'decoder.layer.').
                                         replace('decode.', 'decoder.layer.').
                                         replace('decode.', 'decoder.layer.').
                                         replace('encode.', 'encoder.layer.').
                                         replace('encode.', 'encoder.layer.').
                                         replace('encode.', 'encoder.layer.').
                                         replace('encode.', 'encoder.layer.').
                                         replace('encode.', 'encoder.layer.')] = fvae_ckpt['model_states']['VAE'].pop(key)


    fvae.load_state_dict(fvae_ckpt['model_states']['VAE'])
dvae=fvae


#dvae_ckpt_path = os.path.join(args.output_dir, config['fvae']['runname'], 'chkpts', config['fvae']['chkptname'])
#dvae_ckpt = torch.load(dvae_ckpt_path,map_location='cpu')['model_states']['net']
#dvae.load_state_dict(dvae_ckpt)

tqdm.write('{}'.format(dvae))
tqdm.write('{}'.format(generator))
tqdm.write('{}'.format(discriminator))


# Put models on gpu if needed
dvae = fvae.to(device)
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, dvae, config
)

# Use multiple GPUs if possible
dvae = nn.DataParallel(dvae)
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer, map_location='cpu'
)

# Logger
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
)

# Distributions
cdist = get_zdist('gauss', args.c_dim, device=device)
zdist = get_zdist(config['z_dist']['type'], args.z_dim,
                  device=device)

n=0

# Save for tests
ntest = 100
ssimAll=[]
mseAll=[]
for runs in range(0, 30):
    x_real, ytest = utils.get_nsamples(data_loader, ntest)
    ztest = zdist.sample((ntest,))
    ctest = cdist.sample((ntest,))
    ztest_ = torch.cat([ztest, ctest], 1)
    #utils.save_images(x_real, path.join(out_dir, 'real.png'))
    # Test generator
    if config['training']['take_model_average']:
        generator_test = copy.deepcopy(generator)
        discriminator_test=copy.deepcopy(discriminator)
        
        checkpoint_io.register_modules(generator_test=generator_test)
        checkpoint_io.register_modules(discriminator_test=discriminator_test)
     
    else:
        generator_test = generator
        discriminator_test = discriminator 
    evaluator = Evaluator(generator_test, zdist, cdist,
                          batch_size, device=device)
    
    z = zdist.sample((batch_size,))
    
    # Evaluator
    dis_evaluator = DisentEvaluator(generator=generator_test, zdist=zdist, cdist=cdist,
                                    batch_size=batch_size, device=device, dvae=dvae)
    
    # Train
    tstart = t0 = time.time()
    it = epoch_idx = -1
    
    # Load checkpoint if existant
    it = checkpoint_io.load('model.pt')
    print('it=',it)
    
    if it != -1:
        logger.load_stats('stats.p')
    
    # Reinitialize model average if needed
    if (config['training']['take_model_average']
            and config['training']['model_average_reinit']):
        update_average(generator_test, generator, 0.)
        
    #x, cs=d
    
    ##Image Image comparison qualitative
    TestImage=np.swapaxes(x_real[0,:,:,:],0,2)
    TestImage=np.array(TestImage)
    
    mu, c_lat=dis_evaluator.predict_latent(x_real.cpu())
    recon_img=dis_evaluator.predict_image(c_lat)
    recon_img=torch.stack(recon_img)
    #recon_img=torch.Tensor(recon_img)
    recon_img=recon_img.add(1).div(2).clamp(0, 1)
    real_img=x_real.add(1).div(2).clamp(0, 1)
    
    
    #Get the images in grid for results display
    fake_name_grid='Z:/COVID-FTP/Rashmi/data/Reconstructed/Fake/fake_grid.png'
    
    grid_img_fake= torchvision.utils.make_grid(recon_img, nrow=2)
    grid_img_fake=grid_img_fake.cpu().detach()
    plt.imshow(color.rgb2gray(grid_img_fake.permute(1, 2, 0)), cmap='viridis')
    plt.imshow((grid_img_fake.permute(1, 2, 0)))
    
    plt.axis('off')
    plt.colorbar()
    
    #plt.savefig(fake_name_grid, bbox_inches='tight')
    
    real_name_grid='Z:/COVID-FTP/Rashmi/data/Reconstructed/Real/real_grid.png'
    grid_img_real= torchvision.utils.make_grid(real_img, nrow=2)
    grid_img_real=grid_img_real.cpu().detach()
    plt.imshow(color.rgb2gray(grid_img_real.permute(1, 2, 0)), cmap='viridis')
    #plt.imshow(color.rgb2gray(grid_img_real.permute(1, 2, 0)),cmap='gray')
    
    #plt.imshow((grid_img_real.permute(1, 2, 0)))
    
    plt.axis('off')
    #plt.savefig(real_name_grid, bbox_inches='tight')
    
    #saving full resolution images
    from PIL import Image 
    import PIL 
    #plt.imshow(TestImage)
    #plt.imshow(np.swapaxes(torch.squeeze(recon_img),0,2))
    plt.axis('off')
    for i in range (0, 10):
        rec_image=recon_img[i]
        real_image=real_img[i]
        n=n+1
        #im=torch.unsqueeze(image,0)
        file_name = os.path.splitext(os.path.basename(str(n)))[0]
        
        real_name='Z:/COVID-FTP/Rashmi/data/Reconstructed/Real/'+ file_name +'.png'
        fake_name='Z:/COVID-FTP/Rashmi/data/Reconstructed/Fake/'+ file_name +'.png'
        
        real=np.swapaxes(real_image,0,2).cpu().detach().numpy()
        fake=np.swapaxes(rec_image,0,2).cpu().detach().numpy()
        #im2 = np.resize(im, (512*512*3)).reshape(512, 512, 3)
        #plt.imshow(im)
        plt.axis('off')
    
        #plt.imsave(real_name,real[:,:,0], cmap='viridis')
        #plt.imsave(fake_name,fake[:,:,1], cmap='viridis')
        
        plt.imsave(real_name,real)
        plt.imsave(fake_name,fake)
        mse = mean_squared_error(real, fake)
        ssim_ = ssim(real[:,:,1], fake[:,:,1], data_range=real.max() - real.min())
        mseAll.append(mse)
        ssimAll.append(ssim_)
mseAll=np.array(mseAll)
ssimAll=np.array(ssimAll)
mean_mse=np.mean(mseAll)
mean_ssim=np.mean(ssimAll)
print('Average MSE=')
print(mean_mse)
print('Average SSIM=')
print(mean_ssim)