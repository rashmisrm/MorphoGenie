# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:17:34 2023

@author: Kevin Tsia
"""


    from skimage.metrics import mean_squared_error
    from skimage.metrics import structural_similarity as ssim
    mse_none = mean_squared_error(real, fake)
    ssim_const = ssim(real[:,:,1], fake[:,:,1],
                      data_range=real.max() - real.min())