sys.path.append("stylegan2-pytorch") 

import numpy as np 
import matplotlib.pyplot as plt

import torch

from plot_images import make_image

import umap 

    
def gen_imgs_from_latents(latents, g_ema):
    
    n = latents.shape[0]

    for i in range(1, n+1, 1): 
        # Generate image 
        img,_ = g_ema([latents[int(i-1),:,:].unsqueeze(0)], input_is_latent=True, noise=noises)
        img = make_image(img)
        plt.imshow(img)
        plt.savefig(f'./imgs/lbfgs/img{"{:03d}".format(int(i))}.png')
        plt.close()  

def gen_umap(latents, exp_name):
    # UMAP analysis 
    latents_np = latents.detach().cpu().flatten(start_dim=1).numpy() # [n,18,512]
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latents_np)
    plt.plot(embedding[:,0],embedding[:,1], '-o')
    plt.title('UMAP Projection of Latent Vectors')
    plt.savefig(f'{exp_name}_UMAP.png')
    plt.close()