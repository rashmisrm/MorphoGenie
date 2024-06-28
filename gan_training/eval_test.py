# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 18:54:36 2021

@author: Kevin Tsia
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:53:58 2021

@author: Kevin Tsia
"""
from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid
from math import sqrt
from gan_training.metrics import inception_score
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import truncnorm
import numpy as np
from gan_training.inputs import get_dataset
import pandas as pd
from skimage import color
import os


def truncated_z_sample(batch_size, z_dim, truncation=1., seed=None):
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim))
    return torch.from_numpy(truncation * values)


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


def generator_postprocess(x):
    return x.add(1).div(2).clamp(0, 1)

def decoder_postprocess(x):
    return torch.sigmoid(x)

gp = generator_postprocess
dp = decoder_postprocess

class Evaluator(object):
    def __init__(self, generator, zdist, batch_size=64,
                 inception_nsamples=60000, device=None, dvae=None, fvae=None):
        self.generator = generator
        self.zdist = zdist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.dvae = fvae

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            print('batch_size',self.batch_size)

            ztest = self.zdist.sample((self.batch_size,))

            samples = self.generator(ztest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def create_samples(self, z):
        self.generator.eval()
        batch_size = z.size(0)

        # Sample x
        with torch.no_grad():
            x = self.generator(z)
        return x


class DisentEvaluator(object):
    def __init__(self, generator, zdist, cdist, batch_size=64,
                 device=None, dvae=None):
        self.generator = generator
        self.zdist = zdist
        self.cdist = cdist
        self.batch_size = batch_size
        self.device = device
        self.dvae = dvae


    @torch.no_grad()
    def save_samples(self, zdist, cdist, out_dir, batch_size=64, N=50000):
        import os
        from torchvision.utils import save_image
        self.generator.eval()

        samples = []
        quotient, remainder = N//batch_size, N%batch_size
        with tqdm(range(quotient)) as pbar:
            for _ in pbar:
                z = self.zdist.sample((batch_size,))
                c = self.cdist.sample((batch_size,))
                z_ = torch.cat([z, c], 1)
                sample = gp(self.generator(z_)).data.cpu()
                samples.append(sample)     
                pbar.set_description('generating samples...')

        if remainder > 0:
            z = self.zdist.sample((remainder,))
            c = self.cdist.sample((remainder,))
            z_ = torch.cat([z, c], 1)
            sample = gp(self.generator(z_)).data.cpu()
            samples.append(sample)     

        samples = torch.cat(samples, 0)
        with tqdm(enumerate(samples)) as pbar:
            for i, sample in pbar:
                path = os.path.join(out_dir, '{}.png'.format(i+1))
                save_image(sample, path, 1, 0)
                pbar.set_description('now saving samples...')

        return None

    @torch.no_grad()
    def create_samples(self, z, c):
        self.generator.eval()
        print(z)
        bs =  z.size(0)
        nrow = ncol = int(sqrt(bs))
        bs = nrow*ncol
        print('nrow=',nrow)
        print('bs=',bs)
        z = z[:bs]
        c = c[:bs]
        # point-wise synthesis of z and c
        print('z=',z.size())
        print('c=',c.size())

        z_ = torch.cat([z, c], 1)
        print('Z_ Tensor=',z_[:0].size())

        print('len of Z_=',len(z_))
        print('len of Z=',len(z))
        print('len of C=',len(c))
        
        #x_point = gp(self.generator(z_))
        x_point = (self.generator(z_))
        x_point = make_grid(x_point, nrow=ncol, padding=2, pad_value=1)
        print('rashmi, len of xpoint is')
      #  print(torch.Size(x_point[1]))
        print('xpoint is:')
        #print(x_point)
      #  im=x_point.cpu()
      #  im=x_point.numpy()
    
      #  im=np.swapaxes(im,0,2)
        

#print(np.size(im))
#array = np.reshape(im, (64, 64))
      #  plt.imsave('generated.jpeg', im)
        # single z
        z_ = torch.cat([z[:1].repeat(bs, 1), c], 1)
        print('SingleZ',z[:1].repeat(bs, 1).size())
        #x_singlez = gp(self.generator(z_))
        x_singlez = gp(self.generator(z_))
        x_singlez = make_grid(x_singlez, nrow=ncol, padding=2, pad_value=1)
        
        # single c
        z_ = torch.cat([z, c[:1].repeat(bs, 1)], 1)
        #x_singlec = gp(self.generator(z_))
        x_singlec = (self.generator(z_))
        x_singlec = make_grid(x_singlec, nrow=ncol, padding=2, pad_value=1)
  
        # 1st column c and 1st row z
        cc = c.view(nrow, ncol, -1).permute(1, 0, 2).contiguous().view(bs, -1)[:ncol].unsqueeze(1).repeat(1, ncol, 1).view(bs, -1)
        zz = z[:ncol].unsqueeze(0).repeat(nrow, 1, 1).view(bs, -1)
        z_ = torch.cat([zz, cc], 1)
        #x_fcfz = gp(self.generator(z_))
        x_fcfz = gp(self.generator(z_))

        x_fcfz = make_grid(x_fcfz, nrow=ncol, padding=2, pad_value=1)

        return x_point[0], x_singlez[0], x_singlec[0], x_fcfz[0]

    @torch.no_grad()
    def traverse_c1z1(self, travN, z=None, c=None, limit=3, ncol=10, dims=-1, save='True', save_type='idgan', TravImRet='False', cmap='inferno'):
        # traverse with a single c and z
        self.generator.eval()
        interpolation = torch.linspace(-limit, limit, ncol)
        if z is None:
            z = self.zdist.sample((1,))
        if c is None:
            c = self.cdist.sample((1,))

        idgan_samples = []
        idgan_samples_p = []
        idgan_samples_p_np=[]
        dvae_samples = []
        dvae_samples_p = []
       # mu_2d_gan= []
      #  mu_2d_dvae=[]
        c_ori = c.clone()
        for c_dim in range(self.cdist.dim):
            if dims != -1 and c_dim not in dims:
                continue

            c = c_ori.clone()
            c_ = c_ori.clone()
            c_zero = torch.zeros_like(c)
            for val in interpolation:
                c[:, c_dim] = val
                z_ = torch.cat([z, c], 1)
                idgan_sample = F.adaptive_avg_pool2d(gp(self.generator(z_)), (64, 64))
                idgan_sample = F.adaptive_avg_pool2d(gp(self.generator(z_)), (64, 64)).data.cpu()

                idgan_samples.append(idgan_sample)
                dvae_sample = dp(self.dvae(c=c, decode_only=True)).data.cpu()
                dvae_samples.append(dvae_sample)
                

                c_zero[:, c_dim] = val
                c_p = c_ + c_zero
                z_p_ = torch.cat([z, c_p], 1)
                idgan_sample_p_np = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (512, 512))
                idgan_sample_p = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (64, 64)).data.cpu()
                
                
                idgan_samples_p_np.append(idgan_sample_p)
                dvae_sample_p = dp(self.dvae(c=c_p, decode_only=True)).data.cpu()
                dvae_samples_p.append(dvae_sample_p)
             
                if save=='True':
                    save_path='./outputs/TraversalLoop/'
                    if  os.path.exists(save_path)==False:
                        os.mkdir(save_path)
                    if  os.path.exists(save_path+str(travN)+'/')==False:
                        os.mkdir(save_path+str(travN)+'/')
                        
                    if save_type=='idgan':
                        imagesave=idgan_sample_p_np    
                        imagesave=imagesave.detach().cpu().numpy()
                        imagesave=imagesave.squeeze(0)
                        if imagesave.shape[0]==1:
                            imagesave=imagesave.squeeze(0)
                        else:
                            imagesave=np.swapaxes(imagesave,0,2)

                    else:
                        imagesave=dvae_sample
                        imagesave=imagesave.detach().cpu().numpy()                
                        imagesave=imagesave.squeeze(0)
                        if imagesave.shape[0]==1:
                            imagesave=imagesave.squeeze(0)
                        else:
                            imagesave=np.swapaxes(imagesave,0,2)  

                    val=val.detach().cpu().numpy()
                    plt.imsave((save_path+str(travN)+'/'+ str(c_dim)+'_'+ str(val)+'.png'),imagesave,cmap='gray')
                    
                idgan_samples_p.append(idgan_sample_p)
                idgan_samples_p_np.append(idgan_sample_p_np)
                
        
        idgan_samples = torch.cat(idgan_samples, dim=0)
        idgan_samples = make_grid(idgan_samples, nrow=ncol, padding=2, pad_value=1)
        plt.figure(figsize=(20,24))
        #plt.imshow(color.rgb2gray(idgan_samples.permute(1, 2, 0)), cmap=cmap)
        plt.imshow((idgan_samples.permute(1, 2, 0)))

        plt.axis('off')
        
        dvae_samples = torch.cat(dvae_samples, dim=0)
        dvae_samples = make_grid(dvae_samples, nrow=ncol, padding=2, pad_value=1)
        #plt.imshow((dvae_samples.permute(1, 2, 0)))
        #plt.axis('off')
        
        idgan_samples_p = torch.cat(idgan_samples_p, dim=0)
        idgan_samples_p = make_grid(idgan_samples_p, nrow=ncol, padding=2, pad_value=1)
        #plt.imshow(color.rgb2gray(idgan_samples_p.permute(1, 2, 0)), cmap=cmap)
        #plt.axis('off')
        
        dvae_samples_p = torch.cat(dvae_samples_p, dim=0)
        dvae_samples_p = make_grid(dvae_samples_p, nrow=ncol, padding=2, pad_value=1)
        #plt.imshow(color.rgb2gray(dvae_samples_p.permute(1, 2, 0)), cmap=cmap)
        #plt.axis('off')
              #  idgan_sample_p = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (512, 512)).data.cpu()
  

        #idgan_samples = torch.cat(idgan_samples, dim=0)
        #dvae_samples = torch.cat(dvae_samples, dim=0)


        #2d_dvae=self.reduce_latent2d(mu_cdim_dvae)

        #idgan_samples = torch.cat(idgan_samples, dim=0)
        #mu_cdim_gan, c_lat_gan=self.predict_latent(idgan_samples)

        #idgan_samples = make_grid(idgan_samples, nrow=ncol, padding=2, pad_value=1)
        #dvae_samples = torch.cat(dvae_samples, dim=0)
        #mu_cdim_dvae, c_lat_vae=self.predict_latent(dvae_samples)

        #dvae_samples = make_grid(dvae_samples, nrow=ncol, padding=2, pad_value=1)
        #idgan_samples_p_np=torch.cat(idgan_samples_p_np, dim=0)
        #idgan_samples_p = torch.cat(idgan_samples_p, dim=0)
        #idgan_samples_p = make_grid(idgan_samples_p, nrow=ncol, padding=2, pad_value=1)
        #dvae_samples_p = torch.cat(dvae_samples_p, dim=0)
        #dvae_samples_p = make_grid(dvae_samples_p, nrow=ncol, padding=2, pad_value=1)

        #x = torch.stack([idgan_samples, dvae_samples, idgan_samples_p, dvae_samples_p])
        #x = make_grid(x, nrow=4, padding=4, pad_value=0)
        #x = torch.stack([idgan_samples_p, dvae_samples_p])
        #x = make_grid(x, nrow=4, padding=4, pad_value=0)

    #x=1
        if TravImRet=='True':
             return idgan_samples_p_np

        return ncol
        #return  x, mu_cdim_dvae, mu_cdim_gan, ncol
 
    
    #Prdict Latent Values
    def predict_latent(self,im):
        c,mu,var=self.dvae(x=im, encode_only=True)
        
        #print('mu=',mu)
        #print('var=',var)
        return mu,c
    def predict_image(self,c_lat):
   #     idgan_sample_p = F.adaptive_avg_pool2d(gp(self.generator(z_)), (64, 64)).data.cpu()
   #     idgan_samples_p.append(idgan_sample_p)
   #     dvae_sample_p = dp(self.dvae(c=c_p, decode_only=True)).data.cpu()
   #     dvae_samples_p.append(dvae_sample_p)
        samples=[]
        decoder='idgan'
        for i  in range(0, len(c_lat)):
            z = self.zdist.sample((1,))
            c = c_lat[i]
            c=c.unsqueeze(0)
            if(decoder=='idgan'):
                z_ = torch.cat([z.cpu(), c.cpu()], 1)

                sample = (self.generator(z_))
            elif(decoder=='dvae'):               
                sample = (self.dvae(c=c, decode_only=True))
            elif(decoder=='idgan-p'):
                val=-3
                c_zero = torch.zeros_like(c)
                c_zero[:, :] = val
                c_p = c_lat + c_zero
                z_p_ = torch.cat([z, c], 1)
                sample = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (64, 64))


                
            sample=sample.squeeze()
            samples.append(sample) 
        return samples

    def reduce_latent2d(self,lat_cdim, dim_red=None):
        #dim_red='umap'
        if dim_red=='umap':
            reducer=umap.UMAP()
            embedding = umap.UMAP().fit_transform(lat_cdim)
            #scaled_lat=StandardScaler().fit_transform(lat_cdim)
            #embedding=reducer.fit_transform(scaled_lat)
     #   elif dim_red=='tsne':
            
        elif dim_red=='pca':
            pca = PCA(n_components=2)
            print('rashmi')

            embedding = pca.fit_transform(lat_cdim)
        #    df['pca-one'] = pca_result[:,0]
        #    df['pca-two'] = pca_result[:,1] 
        #    df['pca-three'] = pca_result[:,2]
        elif dim_red=='phate':
            import phate
            phate_op=phate.PHATE()
            embedding=phate_op.fit_transform(lat_cdim)
        return embedding
        
    def normalize_matrix(self, mat):
        Mat=pd.DataFrame(mat)
        for column in Mat.columns:
            Mat[column] = (Mat[column] - Mat[column].min()) / (Mat[column].max() - Mat[column].min())    
        return Mat
    
    def ComputeCorr(self,FeatSel, code, bottleneck):
        
        CorrC = []
        CorrR = []
        corrVal=[]
            ##0-Area 1-Volume 40-DMD 45-peak phase 54-DC2 59-Fit text mean  60-FT Var 65- QPEnt mean 66-QP Ent Var
            
            #Cell Cycle  - Area =0, DMD-15, FT Var-19 QPEntVar-29
        #FeatNo=['0', '1','14','15', '18', '19', '23', '28', '29']
        n1=np.shape(FeatSel)
        from scipy.stats.stats import pearsonr   
        for i in range (0,n1[1]):
            for j in range(0,bottleneck):
                print(j)
                import scipy
                Num=scipy.stats.spearmanr(FeatSel[:,i],code[:,j])
                corr=float(Num[0])
                CorrR.append(corr)
            CorrC.append(CorrR)
            CorrR = []
        return CorrC
    
    
    def GateCells(self, df_F, xmin,xmax,ymin,ymax):
        Gated=[] 
        n=0
        for fn in df_F.FileNames:
            
            if((df_F.col1[n] >= xmin) and (df_F.col1[n] <=xmax) and (df_F.col2[n] >=ymin) and (df_F.col2[n] <= ymax)):
                Gated.append(fn)
            
            n=n+1
                
        return Gated
            
    def saveImages(file_path):
        real_samples=[]
        real_samples = x_real
        with tqdm(enumerate(real_samples)) as pbar:
             for i, real_samples  in pbar:
                     path = os.path.join(out_dir,'Real/' ,'{}.png'.format(i+1))
                     print('sampleSize=',real_samples.size())
                     real_samples=real_samples.add(1).div(2).clamp(0, 1)
        #  save_image(sample, path, 1, 0)
                     real_samples=real_samples.numpy()
                     real_samples=np.swapaxes(real_samples,0,2)
                     plt.imsave(path,real_samples)
         # save_image(sample, path, 1)
          
                     pbar.set_description('now saving samples...')
            
            
        def predict_latent(self,im):
            c,mu,var=self.dvae(x=im, encode_only=True)
            
            #print('mu=',mu)
            #print('var=',var)
            return mu,c
    def DisentMetric(self,c_lat):
       #     idgan_sample_p = F.adaptive_avg_pool2d(gp(self.generator(z_)), (64, 64)).data.cpu()
       #     idgan_samples_p.append(idgan_sample_p)
       #     dvae_sample_p = dp(self.dvae(c=c_p, decode_only=True)).data.cpu()
       #     dvae_samples_p.append(dvae_sample_p)
        if z is None:
            z = self.zdist.sample((1,))
        if c is None:
            c = self.cdist.sample((1,))

        idgan_samples = []
        idgan_samples_p = []
        dvae_samples = []
        dvae_samples_p = []
       # mu_2d_gan= []
      #  mu_2d_dvae=[]
        c_ori = c.clone()
        for c_dim in range(self.cdist.dim):
            if dims != -1 and c_dim not in dims:
                continue

            c = c_ori.clone()
            c_ = c_ori.clone()
            c_zero = torch.zeros_like(c)
            for val in interpolation:
                c[:, c_dim] = val
                z_ = torch.cat([z, c], 1)
                idgan_sample = F.adaptive_avg_pool2d(gp(self.generator(z_)), (64, 64))
              #  idgan_sample = F.adaptive_avg_pool2d(gp(self.generator(z_)), (512, 512)).data.cpu()

                idgan_samples.append(idgan_sample)
                dvae_sample = dp(self.dvae(c=c, decode_only=True)).data.cpu()
                dvae_samples.append(dvae_sample)
                

                c_zero[:, c_dim] = val
                c_p = c_ + c_zero
                z_p_ = torch.cat([z, c_p], 1)
                idgan_sample_p_np = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (512, 512))
                idgan_sample_p = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (64, 64))
                idgan_samples_p.append(idgan_sample_p)
                dvae_sample_p = dp(self.dvae(c=c_p, decode_only=True)).data.cpu()
                dvae_samples_p.append(dvae_sample_p)
                if save=='True':
                    save_path='./outputs/TraversalLoop/'+str(travN)+'/'
                    val=val.detach().cpu().numpy()
                    #idgan_sample_p_np=idgan_sample_p_np.detach().cpu().numpy()
                    dvae_sample=dvae_sample.detach().cpu().numpy()

                    #idgan_sample_p_np=np.rollaxis(idgan_sample_p_np,3,1).reshape(512,-1,512)
                    dvae_sample=np.rollaxis(dvae_sample,3,1).reshape(512,-1,512)

                    #idgan_sample_p_np=np.swapaxes(idgan_sample_p_np,1,2)
                    dvae_sample=np.swapaxes(dvae_sample,1,2)

                    plt.imsave((save_path+ str(c_dim)+'_'+ str(val)+'.png'),dvae_sample)
              #  idgan_sample_p = F.adaptive_avg_pool2d(gp(self.generator(z_p_)), (512, 512)).data.cpu()
  
