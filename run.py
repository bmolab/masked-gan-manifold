import sys
import os 

sys.path.append("stylegan2-pytorch") 

import argparse
import numpy as np 
from numpy import inf

import torch
from torch import optim

from tqdm import tqdm

import lpips
from model import Generator

from PIL import Image 
import matplotlib.pyplot as plt 
import umap 


def make_image(tensor):
    tensor = (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )
    tensor = Image.fromarray(tensor[0])
    return tensor
 
    
def gen_imgs_from_latents(latents, g_ema, noises, exp_name):
    
    n = latents.shape[0]
    
    if not os.path.exists(f'{exp_name}_imgs'):
        os.makedirs(f'{exp_name}_imgs')
    
    for i in range(1, n+1, 1): 
        # Generate image 
        img,_ = g_ema([latents[int(i-1),:,:].unsqueeze(0)], input_is_latent=True, noise=noises)
        img = make_image(img)
        img.save(f'./{exp_name}_imgs/img{i:03d}.png') 
        
        
def gen_umap(latents, exp_name):
    # UMAP analysis 
    latents_np = latents.detach().cpu().flatten(start_dim=1).numpy() # [n,18,512]
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latents_np)
    plt.plot(embedding[:,0],embedding[:,1], '-o')
    plt.title('UMAP Projection of Latent Vectors')
    plt.savefig(f'{exp_name}_UMAP.png')
    plt.close()


def whole_loss(latent, g_ema, noises, args, input_is_latent=True):
    
    whole_latent = latent_in.detach() #  [1,18,512]
    whole_latent[:,args.layer_start:args.layer_end,:] = latent
    
    img_gen, _ = g_ema([whole_latent], input_is_latent=input_is_latent, noise=noises)

    img_dim = img_gen.shape[2]

    mask_left = args.mask_left
    mask_right = args.mask_right
    mask_top = args.mask_top                        # Top border of mask region (also height of top unmask region) 
    mask_bottom = args.mask_bottom                     # Bottom border of mask region 
    unmask_bottom = img_dim-mask_bottom      # Height of bottom unmask region 
    
    mask_height = mask_bottom - mask_top
    mask_width = mask_right - mask_left 
    unmask_area = (img_dim**2) - (mask_height * mask_width)
    
    # Mask loss 
    gen_mask = img_gen[:,:,mask_top:mask_bottom,mask_left:mask_right]
    targ_mask = targ_img.detach().clone()[:,:,mask_top:mask_bottom,mask_left:mask_right]
    mask_loss = torch.sqrt((percept(gen_mask, targ_mask).sum() - args.offset)**2)
    
    # Unmask loss 
    upper = (img_dim*mask_top)/(unmask_area) * (percept(img_gen[:,:,:mask_top,:], targ_img.detach()[:,:,:mask_top,:]).sum()) 
    lower = (img_dim*unmask_bottom)/(unmask_area) * percept(img_gen[:,:,mask_bottom:,:], targ_img.detach()[:,:,mask_bottom:,:]).sum()   
    left = (mask_height*mask_left)/unmask_area * percept(img_gen[:,:,mask_top:mask_bottom,:mask_left], targ_img.detach()[:,:,mask_top:mask_bottom,:mask_left]).sum()
    
    right = (mask_height*(img_dim-mask_right))/unmask_area * (percept(img_gen[:,:,mask_top:mask_bottom,mask_right:], targ_img.detach()[:,:,mask_top:mask_bottom,mask_right:]).sum()) 
    
    unmask_loss = upper + lower + left + right
    
    return args.gamma * unmask_loss, args.gamma * mask_loss


def spring_loss(latents, layer_start, layer_end):      
    # Create block of latent vectors 1:n
    from_offset = latents[1:,layer_start:layer_end,:]

    # Create block of latent vectors 0:n-1
    from_zero = latents[:n-1,layer_start:layer_end,:]

    lengths = (from_offset - from_zero)**2

    lengths = torch.sum(lengths, axis=2)

    lengths = torch.sum(lengths, axis=1)

    # First spring
    lengths = torch.sqrt(torch.sum(torch.sum((from_offset - from_zero)**2, axis=2), axis=1))
    energies = (lengths - anim_speed)**2
    tot_energy = torch.sum(energies, axis=-1)

    # Secondary stiffener spring
    from_offset = latents[2:,layer_start:layer_end,:]
    from_zero = latents[:n-2,layer_start:layer_end,:]        

    s_lengths = torch.sqrt(torch.sum(torch.sum((from_offset - from_zero)**2, axis=2), axis=1))
    s_energies = (s_lengths - s_anim_speed)**2
    s_tot_energy = torch.sum(s_energies, axis=-1)

    return args.alpha * tot_energy, args.beta * s_tot_energy

    
# Use for generating animation along valley floor in w space 
if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = 'cuda'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--latent_seed', type=str, default=None, 
                        help='name of initial seed latent vector. If none is given, one will be randomly generated.')
    parser.add_argument('--checkpoint', type=str, default='stylegan2-ffhq-config-f.pt', 
                        help='name of the checkpoint file.')
    parser.add_argument('--num_vec', type=int, default=36, 
                        help='number of vectors to use within the method.')
    parser.add_argument('--rand_seed', type=int, default=np.random.randint(0,1000), 
                        help='random seed used for generating the seed latent vector and noise used by the generator.')
    parser.add_argument('--mask_left', type=int, default=300, 
                        help='left vertical border of mask region.')
    parser.add_argument('--mask_right', type=int, default=730, 
                        help='right vertical border of mask region.')
    parser.add_argument('--mask_top', type=int, default=680, 
                        help='top horizontal border of mask region.')
    parser.add_argument('--mask_bottom', type=int, default=790, 
                        help='bottom horizontal border of mask region.')
    
    parser.add_argument('--layer_start', type=int, default=2, choices=range(0, 17), 
                        help='Starting layer in W+ to optimize.')
    parser.add_argument('--layer_end', type=int, default=8, choices=range(1, 18), 
                        help='Ending layer in W+ to optimize.')


    parser.add_argument('--alpha', type=float, default=0.3, 
                        help='weight of the primary spring loss term.') # len
    parser.add_argument('--beta', type=float, default=0.1, 
                        help='weight of the secondary spring loss term.') # s_len
    parser.add_argument('--gamma', type=float, default=0.5, 
                        help='weight of L_X loss term.') # L
    parser.add_argument('--prim_dist', type=float, default=4.0, 
                        help='value of sigma, the distance between vectors in the primary spring loss term.')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='number of optimization epochs, will stop early if gradients become too small.')
    parser.add_argument('--offset', type=float, default=0.25, 
                        help='value of c, offset value for the mask region.')

    parser.add_argument('--exp_name', type=str, default='masked_gan_exp', 
                        help='the name of the experiment, used to name the returned files.')
    parser.add_argument('--gen_imgs', type=bool, default=True, 
                        help='if True, generates and saves the images corresponding to the vectors in a directory called {exp_name}_imgs.')
    parser.add_argument('--gen_umap', type=bool, default=True, 
                        help='if True, saves plot of UMAP projection of the resulting vectors in a file named {exp_name}_UMAP.png within the current directory.')

    args = parser.parse_args()
    
    torch.manual_seed(args.rand_seed)
    print(f'random seed: {args.rand_seed}')

    n = args.num_vec  
    anim_speed = args.prim_dist 
    s_anim_speed = anim_speed * 2  
    layer_start = args.layer_start
    layer_end = args.layer_end 
    
    # Initialize generator 
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(args.checkpoint)['g_ema'], strict=False)
    g_ema.eval()

    if torch.cuda.is_available():
        g_ema = g_ema.to(device)

    # Create noise vector 
    noises = g_ema.make_noise()        
    for noise in noises:
        noise.requires_grad = True

    # Load initial seed latent vector
    if args.latent_seed is None:
        # If no seed vector provided, generate a random one 
        sample_z = torch.randn(1, 512, device=device)
        _ = g_ema([sample_z], noise=noises, extract_w=True)
        latent_in = torch.load('rand_w_seed.pt')

    else:
        latent_in = torch.load(args.latent_seed)  

    
    # Generate the original image to use as target 
    targ_img, _ = g_ema([latent_in], input_is_latent=True, noise=noises)

    # Initialize perceptual loss 
    percept = lpips.LPIPS(net='alex')
    percept.cuda()


    ''' 
    Create batch of n latent vectors, all initialized as the same feature vector - shape: [n, 18, 512]
    Add a bit of noise to all but one of the vectors (that one vector is the original anchor point)
    Noise is only added to [:, layer_start:layer_end, :] for these vectors as the default layers most 
    correspond to local features during the generation the image. 
    ''' 
    latents = latent_in.repeat(n,1,1).detach().clone() # [n,18,512]
    
    perturb = torch.zeros(n,18,512) 
    noise = (torch.randn(6,512) * 0.001).unsqueeze(0).repeat(n-1,1,1)
    noise_weights = torch.randn(n-1,1,1)         
    perturb[1:,layer_start:layer_end,:] = noise * noise_weights 
    
    if torch.cuda.is_available():
        latents = latents.to(device)
        latents += perturb.to(device)

    latents.requires_grad = True 
        
    # Optimization
    optimizer = optim.LBFGS([latents], history_size=20, max_iter=8)
    
    # Variables to store each epoch's loss values for printing 
    L_store = 0.0
    length_store = 0.0
    s_length_store = 0.0
    
    pbar = tqdm(range(args.epochs))
    
    def closure():
        optimizer.zero_grad()

        # Calculate and compute grad for primary, secondary, and repel loss   
        len_loss, s_len_loss = spring_loss(latents, layer_start, layer_end)
        length = len_loss + s_len_loss  
        length.backward() 

        '''
        Given the large memory requirement when using lpips loss, we calculate the gradients of each 
        vector of shape [1,6,512] within latents separately. The gradients of each vector are then 
        added together before calling .backwards() 
        '''
        # Create split copy of latent vectors  
        latents_split = latents[:,layer_start:layer_end,:].detach().clone().split(1,dim=0)
        grad = []     # Store .grad of each split vector to be reassembled later  
        epoch_L = 0    # Store loss values for plotting 

        for _tensor in latents_split:           
            # For each view of latent vectors, calculate its loss and store its .grad 
            _tensor.requires_grad = True # [1,6,512]
            unmask_l, mask_l = whole_loss(_tensor, g_ema, noises, args)
            L = unmask_l + mask_l
            L.backward()

            grad_mat = torch.zeros_like(latent_in) # [1,18,512]
            grad_mat[:,layer_start:layer_end,:] = _tensor.grad  
            grad.append(grad_mat) # list of [1,18,512]

            # Accum view loss to get L loss on all latent vectors 
            epoch_L += L.item()                       

        # Concat the .grad for all views and add to latents.grad 
        latents.grad += torch.cat(grad,dim=0)

        # Store loss values in buffer            
        global L_store
        global length_store
        global s_length_store

        L_store=(epoch_L/n)
        length_store=(len_loss.item()/n)
        s_length_store=(s_len_loss.item()/n)

        # Add all losses and return 
        loss = epoch_L + length
        return loss
    
    for i in pbar:  
    
        # Optimize 
        optimizer.step(closure)
        
        pbar.set_description((f'L: {L_store:.4f}; len: {length_store:.4f};'
                              f's_len: {s_length_store:.4f}'))
        
        # If gradients become NaN, go back to previous epoch and break  
        if torch.isnan(latents.grad.norm()).any(): 
            latents = prev_latents 
            break 
        
        prev_latents = latents.clone().detach()
    
    
    # Save latents 
    LATENTS_PATH = f'{args.exp_name}_latents.pt'
    torch.save(latents, LATENTS_PATH)
    
    # Generate images 
    if args.gen_imgs:
        gen_imgs_from_latents(latents, g_ema, noises, args.exp_name)
    
    # Generate UMAP projection 
    if args.gen_umap: 
        gen_umap(latents, args.exp_name)
        
    
    
