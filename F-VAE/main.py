"""main.py"""

import argparse
import numpy as np
import torch
import time
from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def main(args):
    net = Solver(args)
    start_time = time.time()
    net.train()
    #net.eval()
    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factor-VAE')

    parser.add_argument('--image_size', default=256, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--test_bs', default=50, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=10, type=float, help='gamma hyperparameter')
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')

    parser.add_argument('--train_model', default='FactorVAE', type=str, help='Train Model Type-VAE or Beta VAE or FactorVAE')
    #parser.add_argument('--dset_dir', default='H:/ProcessedData/', type=str, help='dataset directory')

    parser.add_argument('--dset_dir', default='C:/Users/Rashmi/WorkingLibs/MorphoGenie-Temp2/MorphoGenie/ProcessedData/', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CCy/', type=str, help='dataset name') #CellCycle/Vero/LC
    parser.add_argument('--datatype', default='rgb', type=str, help='dataset name')
    parser.add_argument('--nc', default=3, type=int, help='Number of Image Channels')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--viz_ll_iter', default=500, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter', default=5000, type=int, help='visdom line data applying iter')
    parser.add_argument('--viz_ra_iter', default=10000, type=int, help='visdom recon image applying iter')
    parser.add_argument('--viz_ta_iter', default=10000, type=int, help='visdom traverse applying iter')

    parser.add_argument('--print_iter', default=500, type=int, help='print losses iter')

    parser.add_argument('--ckpt_dir', default='Z:/COVID-FTP/Rashmi/FactorVAECheckpoints/checkpoints', type=str, help='checkpoint directory')
    #parser.add_argument('--ckpt_dir', default='C:/Users/Rashmi/WorkingLibs/idgan-master_Jan/Models/checkpoints', type=str, help='checkpoint directory')

    parser.add_argument('--ckpt_load', default='20000', type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='checkpoint save iter')
    parser.add_argument('--output_dir', default='Z:/COVID-FTP/Rashmi/FactorVAECheckpoints/outputs', type=str, help='output directory')

    #parser.add_argument('--output_dir', default='C:/Users/Rashmi/WorkingLibs/idgan-master_Jan/Models/outputs', type=str, help='output directory')
    parser.add_argument('--output_save', default=True, type=str2bool, help='whether to save traverse results')
    parser.add_argument('--folder', default='LC', type=str, help='dataset name')
    args = parser.parse_args()

    main(args)

