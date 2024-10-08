import yaml
from torch import optim
from gan_training.models import generator_dict, discriminator_dict, dvae_dict, fvae_dict
from gan_training.train import toogle_grad
import os

# DEFAULT_CONFIG = path.join(path.dirname(__file__), 'configs/default.yaml')


def load_config(path):
    # with open(DEFAULT_CONFIG, 'r') as f:
    #     config = yaml.full_load(f)
    with open(path, 'r') as f:
        config = yaml.full_load(f)
    # config.update(config_new)
    return config

def build_models(image_size, c_dim, nc, z_dims):
    # Get classes
    
    config_path = os.path.join('./configs', 'cells_650.yaml')
    config = load_config(config_path)
    
    gen_name='resnet3'
    disc_name='resnet3'
    dVAE_name='FactorVAE3'
    Generator = generator_dict[config['generator']['name']]
    Discriminator = discriminator_dict[config['discriminator']['name']]
    dVAE = dvae_dict[config['dvae']['name']]
    #dVAE = dvae_dict[dVAE_name]

    if image_size==64:
        fVAE= fvae_dict['FactorVAE64']
        
    elif image_size==128:
        fVAE= fvae_dict['FactorVAE128']

    elif image_size==256:
        fVAE= fvae_dict['FactorVAE256']
    # Build models
    '''dvae = dVAE(
        c_dim=c_dim,
        nc=nc,
        infodistil_mode=True
    )'''
    fvae = fVAE(
        c_dim=c_dim,
        nc=nc,
        infodistil_mode=True
    )
    generator = Generator(
        z_dim= z_dims + c_dim,
        size=image_size,
        nc=nc,
        **config['generator']['kwargs']
    )
    discriminator = Discriminator(
        disc_name,
        size=image_size,
        nc=nc,
        **config['discriminator']['kwargs']
    )

    return fvae, generator, discriminator


def build_optimizers(generator, discriminator, dvae, config):
    optimizer = config['training']['optimizer']
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    equalize_lr = config['training']['equalize_lr']

    toogle_grad(generator, True)
    toogle_grad(discriminator, True)

    if equalize_lr:
        g_gradient_scales = getattr(generator, 'gradient_scales', dict())
        d_gradient_scales = getattr(discriminator, 'gradient_scales', dict())

        g_params = get_parameter_groups(generator.parameters(),
                                        g_gradient_scales,
                                        base_lr=lr_g)
        d_params = get_parameter_groups(discriminator.parameters(),
                                        d_gradient_scales,
                                        base_lr=lr_d)
    else:
        g_params = generator.parameters()
        d_params = discriminator.parameters()

    # Optimizers
    if optimizer == 'rmsprop':
        g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0., 0.99), eps=1e-8)
        d_optimizer = optim.Adam(d_params, lr=lr_d, betas=(0., 0.99), eps=1e-8)
    elif optimizer == 'sgd':
        g_optimizer = optim.SGD(g_params, lr=lr_g, momentum=0.)
        d_optimizer = optim.SGD(d_params, lr=lr_d, momentum=0.)

    return g_optimizer, d_optimizer


def build_lr_scheduler(optimizer, config, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_anneal_every'],
        gamma=config['training']['lr_anneal'],
        last_epoch=last_epoch
    )
    return lr_scheduler


# Some utility functions
def get_parameter_groups(parameters, gradient_scales, base_lr):
    param_groups = []
    for p in parameters:
        c = gradient_scales.get(p, 1.)
        param_groups.append({
            'params': [p],
            'lr': c * base_lr
        })
    return param_groups
