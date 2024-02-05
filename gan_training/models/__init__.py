from gan_training.models import (
    resnet, resnet2, resnet3, dvae
)

dvae_dict = {
    'BetaVAE_H': dvae.BetaVAE_H,
}

fvae_dict = {
    'FactorVAE64': dvae.FactorVAE64,
    'FactorVAE128': dvae.FactorVAE128,
    'FactorVAE256': dvae.FactorVAE256,

}

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'resnet3': resnet3.Generator,
    'dvae_dec': dvae.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'resnet3': resnet3.Discriminator,
    'dvae_enc': dvae.Discriminator,
}
