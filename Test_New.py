import random
import numpy as np
import argparse
import os
from os import path
from tqdm import tqdm
import time
import copy
import torch
from torch import nn
from gan_training.utils import return_data_test
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval_test import DisentEvaluator, Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)

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
parser.add_argument('--bs', default=25, type=int, help='Batch size')
parser.add_argument('--reg_param', default=-1, type=float, help='R1 regularization parameter')
parser.add_argument('--w_info', default=-1, type=float, help='weighting constant on ID Loss')
parser.add_argument('--mi', '--max_iter', default=-1, type=int, help='Max training iteration')
parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')

parser.add_argument('--image_size', default=256, type=int, help='Batch size')  # Generator Image size
parser.add_argument('--dset_dir', default='F:/Rashmi/idgan-master/ProcessedData/', type=str, help='dataset directory')
parser.add_argument('--dataset', default='CPA/', type=str, help='dataset name') #CellCycle/Vero/LC
parser.add_argument('--datatype', default='cpa', type=str, help='dataset name')
#parser.add_argument('--img_size', default=128, type=int, help='Image synthesis size')##This is the image size for which the autoencoder is designed to

parser.add_argument('--nc', default=3, type=int, help='Number of Image channels')

parser.add_argument('--c_dim', default=10, type=int, help='Image synthesis size')
parser.add_argument('--z_dim', default=256, type=int, help='Image synthesis size')

parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda')
parser.add_argument('--seed', default=1, type=int, help='Random Seed')
parser.add_argument('--infodistil_mode', default=True, help='Infodistill')
args = parser.parse_args()


seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

config_path = os.path.join(args.config_dir, args.config)
config = load_config(config_path)
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# = = = = = Customized Configurations = = = = = #
out_dir = os.path.join(args.output_dir, args.name)
config['training']['out_dir'] = out_dir
if args.nf > 0:
    config['generator']['kwargs']['nfilter'] = args.nf 
    config['discriminator']['kwargs']['nfilter'] = args.nf 
if args.bs > 0:
    config['training']['batch_size'] = args.bs
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
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']

checkpoint_dir = path.join(out_dir, 'chkpts')

# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")

train_dataset = get_dataset(
    name=config['data']['type'],
    data_dir=args.dset_dir,
    size=args.image_size,
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
)

        
# Create models
dvae, fvae, generator, discriminator = build_models(config, args)
data_loader = return_data_test(args)



selectVAE='fvae'

if selectVAE=='dvae':
    dvae_ckpt_path = os.path.join(args.output_dir, args.dvae_name, 'chkpts', config['dvae']['chkptname'])

    dvae_ckpt = torch.load(dvae_ckpt_path)['model_states']['net']
    #for key in list(dvae_ckpt.keys()):
    #    dvae_ckpt[key.replace('decoder.', 'module.decoder.'). replace('encoder.', 'module.encoder.')] = dvae_ckpt.pop(key)

    dvae.load_state_dict(dvae_ckpt)
else:

    fvae_ckpt_path = os.path.join(args.output_dir, args.dvae_name,'chkpts', config['fvae']['chkptname'])

    fvae_ckpt = torch.load(fvae_ckpt_path)

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
tqdm.write('{}'.format(dvae))
tqdm.write('{}'.format(generator))
tqdm.write('{}'.format(discriminator))

# Put models on gpu if needed
dvae = dvae.to(device)
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, dvae, config
)

# Use multiple GPUs if possible
dvae = nn.DataParallel(dvae)
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
)

# Logger
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
)

# Distributions
cdist = get_zdist('gauss',args.c_dim, device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)


# Save for tests
ntest = batch_size
#x_real, ytest = utils.get_nsamples(train_loader, ntest)
ztest = zdist.sample((ntest,))
ctest = cdist.sample((ntest,))
ztest_ = torch.cat([ztest, ctest], 1)
#utils.save_images(x_real, path.join(out_dir, 'real.png'))

# Test generator
if config['training']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Evaluator
dis_evaluator = DisentEvaluator(generator=generator_test, zdist=zdist, cdist=cdist,
                                batch_size=batch_size, device=device, dvae=dvae)
data_loader = return_data_test(args)

# Train
tstart = t0 = time.time()
it = epoch_idx = -1

# Load checkpoint if existant
it = checkpoint_io.load('model.pt')
if it != -1:
    logger.load_stats('stats.p')

# Reinitialize model average if needed
if (config['training']['take_model_average']
        and config['training']['model_average_reinit']):
    update_average(generator_test, generator, 0.)

# Learning rate anneling
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

# Trainer
trainer = Trainer(
    dvae, generator, discriminator, g_optimizer, d_optimizer,
    reg_param=config['training']['reg_param'],
    w_info = config['training']['w_info']
)

# Training loop
tqdm.write('Start training...')
pbar = tqdm(total=max_iter)
if it > 0:
    pbar.update(it)
mu_real_Dump=[]
mu_gen_Dump=[]
mu_real_All=[]
mu_gen_All=[]
label1=[]
label2=[]
travN_MainDir='./outputs/TraversalLoop/'

for x_true, path, label in data_loader:
    epoch_idx += 1

    #for x_real, _ in train_loader:
    
    x_true=x_true.to(device)

    mu_real, logvar, c_real = fvae(x_true, encode_only=True)
    print(c_real)
    
    recon_img=dis_evaluator.predict_image(c_real)
    recon_img=torch.stack(recon_img)
    mu_gen, logvar_gen, c_gen=fvae(recon_img, encode_only=True)
    
    mu_real_All.append(mu_real.detach().cpu())
    mu_gen_All.append(mu_gen.detach().cpu())
    
    
    mu, c_lat=dis_evaluator.predict_latent(x_true)
    Traverse=1
    if Traverse==True:
        if epoch_idx < 50:
            
            travN =  epoch_idx
            travN_dir = os.path.join(travN_MainDir, str(travN))
            #print('checkpoint_dir',checkpoint_dir)
            # Create missing directories
            if not os.path.exists(travN_dir):
                os.makedirs(travN_dir)
                
            c_lat_sample=c_lat
            z_sample=ztest[0]
            #c_lat_sample=c_lat_sample.unsqueeze(0)
            z_sample=z_sample.unsqueeze(0)
            travN
            ncol=dis_evaluator.traverse_c1z1(travN, z_sample, c_lat_sample, save='True', save_type='idgan', TravImRet='True', cmap='viridis')

## Sorting the feature tables according to order generated in the ImageGenerator 
    label1.append(label)
    label2.append(path)

label1=np.hstack(label1)
label2=np.hstack(label2)

mu_real_All=np.vstack(mu_real_All)
embedding=dis_evaluator.reduce_latent2d(dis_evaluator.normalize_matrix(mu_real_All),dim_red=('umap'))
df_F=[]

#data_real=np.hstack((embedding, labelAll[:]))
df1 = pd.DataFrame(embedding, columns=['col1','col2'])
df2 = pd.DataFrame(label1, columns=['ClassLabels1'])
df3 = pd.DataFrame(label2, columns=['ClassLabels2'])
df_F=pd.concat([df1, df2, df3],axis=1)

plt.figure(figsize=(10,10))
sns.set(font_scale = 1)
plt.scatter(embedding[:,0], embedding[:,1], s=150)
plt.legend(markerscale=4, fontsize=50)

mu_gen_A=torch.vstack(mu_gen_All)
mu_gen_A=mu_gen_A.detach().numpy()

embedding=dis_evaluator.reduce_latent2d(dis_evaluator.normalize_matrix(mu_gen_A))
df_F=[]
 
#data_real=np.hstack((embedding, labelAll[:]))
df1 = pd.DataFrame(embedding, columns=['col1','col2'])
df2 = pd.DataFrame(label1, columns=['ClassLabels1'])
df3 = pd.DataFrame(label2, columns=['ClassLabels2'])
df_F=pd.concat([df1, df2, df3],axis=1)



plt.figure(figsize=(10,10))
sns.set(font_scale = 1)
sns.scatterplot(df1)
plt.legend(markerscale=4, fontsize=50)
### Reconstruction

### Latent traversals for Interpretability

mu, c_lat=dis_evaluator.predict_latent(x_true)
#looping traversals
#for travN in range(1, len(x_real)):
for travN in range(1, 25):

    travN_dir = os.path.join(travN_MainDir, str(travN))
    #print('checkpoint_dir',checkpoint_dir)
    # Create missing directories
    if not os.path.exists(travN_dir):
        os.makedirs(travN_dir)
        
    c_lat_sample=c_lat[travN]
    z_sample=ztest[0]
    c_lat_sample=c_lat_sample.unsqueeze(0)
    z_sample=z_sample.unsqueeze(0)
    travN
    ncol=dis_evaluator.traverse_c1z1(travN, z_sample, c_lat_sample, save='True', TravImRet='False')



sns.set(font_scale=1)
dis_evaluator.DisentMetric(c_lat_sample)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
# Split dataset into training set and test set
plt.grid('off')
X_train, X_test, y_train, y_test = train_test_split(mu_real_All, label1, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf.fit(X_train, y_train)  # celltype
df_X=pd.DataFrame(mu_real_All)
pd.Series(clf.feature_importances_, index=df_X.columns).plot.bar(color='steelblue', figsize=(12, 6))
plt.show()
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


sns.set(color_codes=True) 
#g1=sns.clustermap(df_X, standard_scale=1, figsize=(30,150),xticklabels=df_X.columns,linewidths=8,tree_kws=dict(linewidths=1))
g1=sns.clustermap(mu_real_All, standard_scale=1, cmap='PiYG_r')
sns.heatmap(mu_real_All,cmap="PiYG_r")


from torchmetrics.functional.classification import multiclass_auroc
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
X=  mu_real_All
X = mu_gen_A
y=label1
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')