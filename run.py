sys.path.append("stylegan2-pytorch") 

import argparse
import numpy as np 
from numpy import inf

import torch
from torch import optim

from tqdm import tqdm

import lpips
from model import Generator

from gen_figures import gen_imgs_from_latents, gen_umap 

torch.manual_seed(0)

if torch.cuda.is_available():
    device = 'cuda'
       
parser = argparse.ArgumentParser()
parser.add_argument('--seed_latent', type=str, default=None, 
                    help='name of initial seed latent vector. If none is given, one will be randomly generated')
parser.add_argument('--num_vec', type=int, default=36, 
                    help='number of vectors to use within the method')
parser.add_argument('--mask_left', type=int, default=300, 
                    help='left vertical border of mask region')
parser.add_argument('--mask_right', type=int, default=730, 
                    help='right vertical border of mask region')
parser.add_argument('--mask_top', type=int, default=680, 
                    help='top horizontal border of mask region')
parser.add_argument('--mask_bottom', type=int, default=790, 
                    help='bottom horizontal border of mask region')

parser.add_argument('--alpha', type=float, default=0.3, 
                    help='weight of the primary spring loss term') # len
parser.add_argument('--beta', type=float, default=0.1, 
                    help='weight of the secondary spring loss term') # s_len
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='weight of L_X loss term') # L
parser.add_argument('--prim_dist', type=float, default=4.0, 
                    help='value of sigma, the distance between vectors in the primary spring loss term')
parser.add_argument('--steps', type=int, default=500, 
                    help='number of optimization epochs, will stop early if gradients become too small')
parser.add_argument('--offset', type=float, default=0.25, 
                    help=='value of c, offset value for the mask region')

parser.add_argument('--exp_name', type=str, default='masked_gan_exp', 
                    help='the name of the experiment, used to name the returned files')
parser.add_argument('--gen_imgs', type=bool, default=True, 
                    help='if True, generates and saves the images corresponding to the vectors in a directory called {exp_name}_imgs')
parser.add_argument('--gen_umap', type=bool, default=True, 
                    help='if True, saves plot of UMAP projection of the resulting vectors in a file named {exp_name}_UMAP.png within the current directory')


args = parser.parse_args()

    
# Initialize generator 
g_ema = Generator(1024, 512, 8)
g_ema.load_state_dict(torch.load('stylegan2-ffhq-config-f.pt')['g_ema'], strict=False)
g_ema.eval()

if torch.cuda.is_available():
    g_ema = g_ema.to(device)

# Create noise vector 
noises = g_ema.make_noise()        
for noise in noises:
    noise.requires_grad = True

# Load initial seed latent vector
if args.seed_latent is None:
    pass
else:
    latent_in = torch.load(args.seed_latent)  

# Generate the original image to use as target 
targ_img, _ = g_ema([latent_in], input_is_latent=True, noise=noises)

# Initialize perceptual loss 
percept = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=device.startswith('cuda'))

def whole_loss(latent, input_is_latent=True):

    mask_left = args.mask_left
    mask_right = args.mask_right
    mask_top = args.mask_top                        # Top border of mask region (also height of top unmask region) 
    mask_bottom = args.mask_bottom                     # Bottom border of mask region 
    unmask_bottom = 1024-mask_bottom      # Height of bottom unmask region 
    
    mask_height = mask_bottom - mask_top
    mask_width = mask_right - mask_left 
    unmask_area = (1024**2) - (mask_height * mask_width)
    
    whole_latent = latent_in.detach() #  [1,18,512]
    whole_latent[:,2:8,:] = latent
    
    img_gen, _ = g_ema([whole_latent], input_is_latent=input_is_latent, noise=noises)
    
    # Mask loss 
    gen_mask = img_gen[:,:,mask_top:mask_bottom,mask_left:mask_right]
    targ_mask = targ_img.detach().clone()[:,:,mask_top:mask_bottom,mask_left:mask_right]
    mask_loss = torch.sqrt((percept(gen_mask, targ_mask).sum() - args.offset)**2)
    
    # Unmask loss 
    upper = (1024*mask_top)/(unmask_area) * (percept(img_gen[:,:,:mask_top,:], targ_img.detach()[:,:,:mask_top,:]).sum()) 
    lower = (1024*unmask_bottom)/(unmask_area) * percept(img_gen[:,:,mask_bottom:,:], targ_img.detach()[:,:,mask_bottom:,:]).sum()   
    left = (mask_height*mask_left)/unmask_area * percept(img_gen[:,:,mask_top:mask_bottom,:mask_left], targ_img.detach()[:,:,mask_top:mask_bottom,:mask_left]).sum()
    
    right = (mask_height*(1024-mask_right))/unmask_area * (percept(img_gen[:,:,mask_top:mask_bottom,mask_right:], targ_img.detach()[:,:,mask_top:mask_bottom,mask_right:]).sum()) 
    
    unmask_loss = upper + lower + left + right
    
    return args.gamma * unmask_loss, args.gamma * mask_loss

# Use for generating animation along valley floor in w space 
if __name__ == '__main__':
    
    n = args.num_vec   # Number of latent vectors 
    anim_speed = args.prim_dist 
    s_anim_speed = anim_speed * 2    
    
    def spring_loss(latents):      
        # Create block of latent vectors 1:n
        from_offset = latents[1:,2:8,:]
        
        # Create block of latent vectors 0:n-1
        from_zero = latents[:n-1,2:8,:]
        
        lengths = (from_offset - from_zero)**2
        
        lengths = torch.sum(lengths, axis=2)
        
        lengths = torch.sum(lengths, axis=1)
        
        # First spring
        lengths = torch.sqrt(torch.sum(torch.sum((from_offset - from_zero)**2, axis=2), axis=1))
        energies = (lengths - anim_speed)**2
        tot_energy = torch.sum(energies, axis=-1)
        
        # Secondary stiffener spring
        from_offset = latents[2:,2:8,:]
        from_zero = latents[:n-2,2:8,:]        
                
        s_lengths = torch.sqrt(torch.sum(torch.sum((from_offset - from_zero)**2, axis=2), axis=1))
        s_energies = (s_lengths - s_anim_speed)**2
        s_tot_energy = torch.sum(s_energies, axis=-1)
        
        return args.alpha * tot_energy, args.beta * s_tot_energy
    
   
    
    # Create batch of n latent vectors, all initialized as the same feature vector - shape: [6, 512, n]
    # Add a bit of noise to all but one of the vectors (that one vector is the original anchor point)
    latents = latent_in.repeat(n,1,1).detach().clone() # [n,18,512]
    
    perturb = torch.zeros(n,18,512) 
    noise = (torch.randn(6,512) * 0.001).unsqueeze(0).repeat(n-1,1,1)
    noise_weights = torch.randn(n-1,1,1)         
    perturb[1:,2:8,:] = noise * noise_weights
    
    latents.requires_grad = True 
    if torch.cuda.is_available():
        latents = latents.to(device)
        latents += perturb.to(device)
        
    # Optimization
    optimizer = optim.LBFGS([latents], history_size=20, max_iter=8)
    
    L_str = 0.0
    length_str = 0.0
    s_length_str = 0.0
    
    pbar = tqdm(range(args.steps))
    
    for i in pbar:  
 
        def closure():
            optimizer.zero_grad()

            # Calculate and compute grad for primary, secondary, and repel loss   
            len_loss, s_len_loss = spring_loss(latents)
            length = len_loss + s_len_loss  
            length.backward() 

            # Create split copy of latent vectors  
            latents_split = latents[:,2:8,:].detach().clone().split(1,dim=0)
            grad = []
            epoch_L = 0    # Store loss values for plotting 
            
            for _tensor in latents_split:           
                # For each view of latent vectors, calculate its loss and store its .grad 
                _tensor.requires_grad = True # [1,6,512]
                unmask_l, mask_l = whole_loss(_tensor)
                L = unmask_l + mask_l
                L.backward()

                grad_mat = torch.zeros_like(latent_in) # [1,18,512]
                grad_mat[:,2:8,:] = _tensor.grad  
                grad.append(grad_mat) # list of [1,18,512]

                # Accum view loss to get L loss on all latent vectors 
                epoch_L += L.item()                       

            # Concat the .grad for all views and add to latents.grad 
            latents.grad += torch.cat(grad,dim=0)
                       
            # Store loss values in buffer            
            global L_str
            global length_str
            global s_length_str
            
            L_str=(epoch_L/n)
            length_str=(len_loss.item()/n)
            s_length_str=(s_len_loss.item()/n)
            
            # Add all losses and return 
            loss = epoch_L + length
            return loss
    
        # Optimize 
        optimizer.step(closure)
    
        pbar.set_description((f'L: {L_str:.4f}; len: {length_str:.4f};'
                              f's_len: {s_length_str:.4f}))
        
        # Check for and save checkpoint 
        if torch.isnan(latents.grad.norm()).any() or i==args.steps-1: 
            PATH = f'./{args.exp_name}_latents.pt'
            torch.save({'latents': prev_latents}, PATH)
            break 
        
        prev_latents = latents.clone().detach()
    
    
    # Generate images and UMAP figure 
    latents = torch.load(f'./{args.exp_name}_latents.pt')['latents']
    if args.gen_imgs:
        gen_imgs_from_latents(latents, g_ema)
    
    if args.gen_umap: 
        gen_umap(latents, args.exp_name)
                                  