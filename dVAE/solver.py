"""solver.py"""

import os
import visdom
from tqdm import tqdm
import torchvision
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import numpy as np
from utils import DataGather, mkdirs, grid2gif
from ops import recon_loss, kl_divergence, permute_dims
from model import FactorVAE1, FactorVAE64, FactorVAE128, Discriminator, FactorVAE256, FactorVAE512
from dataset import return_data, test_return_data
import matplotlib.pyplot as plt

class Solver(object):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.pbar = tqdm(total=self.max_iter)

        # Data
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.datatype = args.datatype
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)
        self.test_data_loader = test_return_data(args)
        #self.train_features, self.train_labels = return_data(args)

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.nc = args.nc

        self.gamma = args.gamma
        self.train_model=args.train_model
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D
        
        self.beta1_E = args.beta1_D
        self.beta2_E = args.beta2_D
        #self.num_Cell_Classes=args.num_Cell_Classes
        #self.num_Batch_Classes=args.num_Batch_Classes

        if args.image_size == 64:
            self.VAE = FactorVAE64(self.z_dim, self.nc).to(self.device)
            self.nc = args.nc
        elif args.image_size == 128:
            self.VAE = FactorVAE128(self.z_dim, self.nc).to(self.device)   #FactorVAE5- 256 FactorVAE4- 128
            self.nc = args.nc
        elif args.image_size == 256:
            self.nc = args.nc
            self.VAE = FactorVAE256(self.z_dim, self.nc).to(self.device)
            
        elif args.image_size == 512:
            self.nc = args.nc
            self.VAE = FactorVAE512(self.z_dim, self.nc).to(self.device)
            
            
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.nets = [self.VAE, self.D]
        
        # Visdom
        self.viz_on = args.viz_on
        self.win_id = dict(D_z='win_D_z', recon='win_recon', kld='win_kld', acc='win_acc', total='win_total')
        self.line_gather = DataGather('iter', 'soft_D_z', 'soft_D_z_pperm', 'recon', 'kld', 'acc', 'vae_loss')
        self.image_gather = DataGather('true', 'recon')
        if self.viz_on:
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_ra_iter = args.viz_ra_iter
            self.viz_ta_iter = args.viz_ta_iter
            #if not self.viz.win_exists(env=self.name+'/lines', win=self.win_id['D_z']):
            self.viz_init()
            self.viz.line(X=np.array([0]),
                      Y=np.array([0]),
                      #env=self.name+'/lines',
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
            self.viz.line(X=np.array([0]),
                      Y=np.array([0]),
                      #env=self.name+'/lines',
                      win=self.win_id['acc'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
            self.viz.line(X=np.array([0]),
                      Y=np.array([0]),
                      #env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))
            
            self.viz.line(X=np.array([0]), 
                      Y=np.array([0]),
                      #env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='Total Loss',))           

        #from visdom import Visdom

        #self.viz = Visdom() 
        #self.viz.line([[0.,0.]], [0], win='Train', opts=dict(title='VAE/Discriminator Loss and Accuracy', 
        #                                                     legend=['soft_D_z', 'soft_D_z_pperm', 'recon', 'kld', 'acc']))
        #self.viz.line([[0.,0.]], [0], win='Test', opts=dict(title='Total Loss and Accuracy', legend=['loss', 'acc']))


        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            self.load_checkpoint()

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)

    def train(self):
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:
                #for x_true2 in self.train_features:
                #x_true1=x_true1.unsqueeze(0)
                #x_true2=x_true2.unsqueeze(0)
                
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu, logvar)

                D_z = self.D(z)
                vae_tc_loss1 = (D_z[:, :1] - D_z[:, 1:]).mean()
                vae_tc_loss = (D_z[ :1] - D_z[ 1:]).mean()
                
                x_true2 = x_true2.to(self.device)

                z_prime = self.VAE(x_true2, no_dec=True)

                #E_z_prime=self.E(z_prime)
                z_prime=z_prime.to(self.device)
                
                #E_loss=Distance(z, z_prime,E_z, E_z_prime)
                #E_loss=Distance(z, z_prime,E_z)
                #E_loss=torch.FloatTensor(E_loss)

                #beta=10
                
                if self.train_model=='VAE':
                    vae_loss =  vae_kld + vae_recon_loss
                elif self.train_model=='BetaVAE':
                    vae_loss =  self.gamma*vae_kld + vae_recon_loss
                else:
                    vae_loss =  vae_kld + self.gamma*vae_tc_loss + vae_recon_loss



                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                # self.optim_VAE.step()

                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))



                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_VAE.step()
                self.optim_D.step()


                if self.global_iter%self.print_iter == 0:
                    self.pbar.write('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                        self.global_iter, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))

                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)
                    self.image_gather.insert(true=x_true1.data.cpu(),
                                             recon=torch.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    #self.viz_latent()
                    self.image_gather.flush()
                    self.traversal_save()

                if self.viz_on and (self.global_iter%self.viz_ll_iter == 0):
                    
                    soft_D_z = F.softmax(D_z, 1)[:, :1].detach()
                    soft_D_z_pperm = F.softmax(D_z_pperm, 1)[:, :1].detach()
                    D_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_pperm < 0.5).sum()).float()
                    D_acc /= 2*self.batch_size
                    self.line_gather.flush()
                    self.line_gather.insert(iter=self.global_iter,
                                            soft_D_z=soft_D_z.mean().item(),
                                            soft_D_z_pperm=soft_D_z_pperm.mean().item(),
                                            recon=vae_recon_loss.item(),
                                            kld=vae_kld.item(),
                                            acc=D_acc.item(),
                                            vae_loss=vae_loss.item())
                    self.visualize_line()

                    #self.viz.line([[soft_D_z, soft_D_z_pperm]], [self.global_iter], win='train', update='append')


                if self.viz_on and (self.global_iter%self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ra_iter == 0):
                    self.image_gather.insert(true=x_true1.data.cpu(),
                                             recon=torch.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.viz_latent()
                    self.image_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ta_iter == 0):
                    if self.dataset.lower() == '3dchairs':
                        self.traversal_save(limit=2, inter=0.5)
                    else:
                        self.traversal_save(limit=4, inter=2/3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def visualize_recon(self):
        outdir = os.path.join('D:/Letitia/FactorVAE-master/outputs/',self.name,str(self.global_iter),'Recon')
        recon = os.path.join(outdir,'Recon')
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            os.makedirs(recon)

        data = self.image_gather.data
        if self.datatype == 'QPI':
            true_image = data['true'][0]
            recon_image = data['recon'][0]
            
            true_image = make_grid(true_image)
            recon_image = make_grid(recon_image)
            sample = torch.stack([true_image, recon_image], dim=0)
            
            outfile = os.path.join(recon,'real-%08d.png'%int(self.global_iter))
            true_image_x=torch.unsqueeze(true_image,1)
            torchvision.utils.save_image(true_image, outfile, nrow=8)
               
            outfile = os.path.join(recon,'recon-%08d.png' % int(self.global_iter))
            recon_image_x=torch.unsqueeze(recon_image,1)
            torchvision.utils.save_image(recon_image, outfile, nrow=8)


        elif self.datatype=='DICxy':   
            true_image = data['true'][0]
            recon_image = data['recon'][0]
            
            true_image_x=true_image[:,0,:,:]
            true_image_y=true_image[:,1,:,:]
            
            recon_image_x = recon_image[:,0,:,:]
            recon_image_y = recon_image[:,1,:,:]
            print('max:',torch.max(recon_image_x))
            print('min:',torch.min(recon_image_x))
            print('max_real:',torch.max(true_image_x))
            print('min_real:',torch.min(true_image_x))


            
            outfile = os.path.join(recon,'real-x%08d.png'%int(self.global_iter))
            true_image_x=torch.unsqueeze(true_image_x,1)
            torchvision.utils.save_image(true_image_x, outfile, nrow=8)
            
            
            outfile = os.path.join(recon,'real-y%08d.png' % int(self.global_iter))
            true_image_y=torch.unsqueeze(true_image_y,1)
            torchvision.utils.save_image(true_image_y, outfile, nrow=8)
            
            
            outfile = os.path.join(recon,'recon-x%08d.png' % int(self.global_iter))
            recon_image_x=torch.unsqueeze(recon_image_x,1)
            torchvision.utils.save_image(recon_image_x, outfile, nrow=8)
            
            
            outfile = os.path.join(recon,'recon-y%08d.png' % int(self.global_iter))
            recon_image_y=torch.unsqueeze(recon_image_y,1)
            torchvision.utils.save_image(recon_image_y, outfile, nrow=8)

        else:
            
            true_image = data['true'][0]
            recon_image = data['recon'][0]
            
            true_image = make_grid(true_image)
            recon_image = make_grid(recon_image)
            sample = torch.stack([true_image, recon_image], dim=0)

            print('max:',torch.max(recon_image))
            print('min:',torch.min(recon_image))
            print('max_real:',torch.max(true_image))
            print('min_real:',torch.min(true_image))

            
            outfile = os.path.join(recon,'real-%08d.png'%int(self.global_iter))
            true_image_x=torch.unsqueeze(true_image,1)
            torchvision.utils.save_image(true_image, outfile, nrow=8)
               
            outfile = os.path.join(recon,'recon-%08d.png' % int(self.global_iter))
            recon_image_x=torch.unsqueeze(recon_image,1)
            torchvision.utils.save_image(recon_image, outfile, nrow=8)
        #self.viz.images(sample, env=self.name+'/recon_image',
                        #opts=dict(title=str(self.global_iter)))
        
    def traversal_save(self, limit=3, inter=2/3, loc=-1):
        traversal =  os.path.join('D:/Letitia/FactorVAE-master/outputs/',self.name,str(self.global_iter),'Traversal')

            
        if not os.path.exists(traversal):
            os.makedirs(traversal)
            
        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit+0.1, inter)    
        fixed_idx=random.sample(range(0,9),1)
        fixed_img = self.data_loader.dataset.__getitem__(fixed_idx[0])[0]
        fixed_img = fixed_img.to(self.device).unsqueeze(0)
        fixed_img_z = encoder(fixed_img)[:, :self.z_dim]
        z_ori=fixed_img_z
        gifs = []
        samples = []
        for row in range(self.z_dim):
            if loc != -1 and row != loc:
                continue
            z_o = z_ori.clone()
            for val in interpolation:
                z=z_o
                z[:, row] = val
                z=z.squeeze()
                sample = torch.sigmoid(decoder(z)).data
                samples.append(sample)
                gifs.append(sample)
        samples = torch.cat(samples, dim=0).cpu()

        
        if self.datatype=='DICxy':            
            Samples_x = samples[:,0,:,:]
            Samples_y = samples[:,1,:,:]
            
            outfile_x = os.path.join(traversal,'Traversal-x%08d.png' % int(self.global_iter))
            outfile_y = os.path.join(traversal,'Traversal-y%08d.png' % int(self.global_iter))
            Samples_x=torch.unsqueeze(Samples_x,1)
            Samples_y=torch.unsqueeze(Samples_y,1)
        
            torchvision.utils.save_image(Samples_x, outfile_x, nrow=len(interpolation))
            torchvision.utils.save_image(Samples_y, outfile_y, nrow=len(interpolation))
        elif self.datatype=='QPI':       
            outfile_x = os.path.join(traversal,'Traversal-%08d.png' % int(self.global_iter))
            torchvision.utils.save_image(samples, outfile_x, nrow=len(interpolation))

        
        else:      
            outfile_x = os.path.join(traversal,'Traversal-%08d.png' % int(self.global_iter))
            torchvision.utils.save_image(samples, outfile_x, nrow=len(interpolation))


    def visualize_line(self):
        import numpy as np
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon = torch.Tensor(data['recon'])
        kld = torch.Tensor(data['kld'])
        D_acc = torch.Tensor(data['acc'])
        soft_D_z = torch.Tensor(data['soft_D_z'])
        soft_D_z_pperm = torch.Tensor(data['soft_D_z_pperm'])
        soft_D_zs = torch.stack([soft_D_z, soft_D_z_pperm], -1)
        vae_loss =torch.Tensor(data['vae_loss'])

        '''self.viz.line(X=iters,
                      Y=soft_D_zs,
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))'''
        self.viz.line(X=np.array([iters]),
                      Y=np.array([recon]),
                      #env=self.name+'/lines',
                      win=self.win_id['recon'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=np.array([iters]),
                      Y=np.array([D_acc]),
                      #env=self.name+'/lines',
                      win=self.win_id['acc'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        self.viz.line(X=np.array([iters]),
                      Y=np.array([kld]),
                      #env=self.name+'/lines',
                      win=self.win_id['kld'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))
        self.viz.line(X=np.array([iters]),
                      Y=np.array([vae_loss]),
                      #env=self.name+'/lines',
                      win=self.win_id['kld'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))
        
    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit+0.1, inter)

        random_img = self.data_loader.dataset.__getitem__(0)[1]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]


        if self.dataset.lower() == 'qpi':
            fixed_idx1 = 1 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 2 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 3 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 4  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                 'random':random_img_z}

        else:
            fixed_idx = 0
            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = torch.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            self.viz.images(samples[:,0,:,:], env=self.name+'/traverse',
                            opts=dict(title=title), nrow=len(interpolation))

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            mkdirs(output_dir)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    
                    if self.dataset=='QPI':

                        save_image(tensor=gifs[i][j].cpu(),
                                   filename=os.path.join(output_dir, '{}_{}_x.jpg'.format(key, j)),
                                   nrow=self.z_dim, pad_value=1)
                    
                    elif self.dataset=='DICxy' :
                    
                        im_x=gifs[j,0,:,:]
                        save_image(tensor=im_x.cpu(),
                                   filename=os.path.join(output_dir, '{}_{}_x.jpg'.format(key, j)),
                                   nrow=self.z_dim, pad_value=1)
                        im_y=gifs[j,1,:,:]
    
                        save_image(tensor=im.cpu(),
                                   filename=os.path.join(output_dir, '{}_{}_y.jpg'.format(key, j)),
                                   nrow=self.z_dim, pad_value=1)

                grid2gif(str(os.path.join(output_dir, key+'*.jpg')),
                         str(os.path.join(output_dir, key+'.gif')), delay=10)

        self.net_mode(train=True)
        
    def viz_latent(self):
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        LatentZAll=[]
        Label1=[]
        Label2=[]
        LabelAll=[]
        #x_true, y= next(iter(self.test_data_loader)) 
        print('Creating .CSV file...')

        for x_true, lab1, lab2 in self.test_data_loader:

            x_true=x_true.to(self.device)
            latent_z = encoder(x_true)[:, :self.z_dim]
            latent_z=latent_z.detach().cpu().numpy()
            LatentZAll.append(latent_z)
            Label1.append(lab1)
            Label2.append(lab2)
        import numpy as np
        import pandas as pd
        LatentZAll=np.vstack(LatentZAll)
        #LatentZAll=torch.tensor(LatentZAll)
        #LatentZAll= LatentZAll.to(self.device)
        Label1=np.hstack(Label1)
        Label2=np.hstack(Label2)
        
        Label1=Label1[:,None]
        Label2=Label2[:,None]
        LabelAll=np.concatenate((Label1, Label2), axis=1)
        LatentZAll=np.squeeze(LatentZAll)
        LatentZAll=np.squeeze(LatentZAll)
        LatentZ=pd.DataFrame(LatentZAll)
        LatentZ.to_csv('./Latent.csv')
        LabelAll=pd.DataFrame(LabelAll)
        LabelAll.to_csv('./Label.csv')


            
            

        

    def viz_init(self):
        zero_init = torch.zeros([1])
        self.viz.line(X=zero_init,
                      Y=torch.stack([zero_init, zero_init], -1),
                      env=self.name+'/lines',
                      win=self.win_id['D_z'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['acc'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))
            


    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))

    def eval(self):
        self.net_mode(train=False)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        out = False
        #while not out:
        #for x_true1, x_true2 in self.data_loader:
            #for x_true2 in self.train_features:
            #x_true1=x_true1.unsqueeze(0)
            #x_true2=x_true2.unsqueeze(0)
            
            #self.global_iter += 1
        self.global_iter=20000
        self.pbar.update(1)

            #x_true1 = x_true1.to(self.device)
            #x_recon, mu, logvar, z = self.VAE(x_true1)
            #vae_recon_loss = recon_loss(x_true1, x_recon)
            #vae_kld = kl_divergence(mu, logvar)

            #D_z = self.D(z)
            #vae_tc_loss1 = (D_z[:, :1] - D_z[:, 1:]).mean()
            #vae_tc_loss = (D_z[ :1] - D_z[ 1:]).mean()
            
            ##z_prime = self.VAE(x_true2, no_dec=True)



        if self.viz_on and (self.global_iter%self.viz_ra_iter == 0):
#            self.image_gather.insert(true=x_true1.data.cpu(),
#                                     recon=torch.sigmoid(x_recon).data.cpu())
            #self.visualize_recon()
            
            #self.image_gather.flush()
            self.viz_latent()
#        if self.viz_on and (self.global_iter%self.viz_ta_iter == 0):
#            if self.dataset.lower() == '3dchairs':
#                self.traversal_save(limit=2, inter=0.5)
#            else:
#                self.traversal_save(limit=4, inter=2/3)

        if self.global_iter >= self.max_iter:
            out = True
            #break

        self.pbar.write("[Eval Finished]")
        self.pbar.close()
