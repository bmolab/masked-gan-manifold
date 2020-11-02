# Mask-Guided Discovery of Semantic Manifolds in Generative Models

**[Mask-Guided Discovery of Semantic Manifolds in Generative Models](https://mengyu.page/files/workshop.pdf)**<br>
  *[Mengyu Yang](https://mengyu.page/),
  [David Rokeby](https://www.cdtps.utoronto.ca/people/directories/all-faculty/david-rokeby),
  [Xavier Snelgrove](https://wxs.ca/)<br>
  BMO Lab for Creative Research, University of Toronto  
in Workshop on Machine Learning for Creativity and Design (NeurIPS), 2020*

A mask-guided, optimization based approach to learning semantic manifolds in StyleGAN2. Given an initial latent vector, this method finds additional vectors corresponding to the same image but with changes localized within the mask region. 

![overview figure](figures/overview.png)

## How Does it Work? 

Starting with an initial generated image and its corresponding latent vector (orange in the figure above), we define a rectangular mask region M over
the image, such as around the mouth. The manifold we seek contains images and their latent vectors (red) that have primarily changed in the mask region but not in the rest of the image. We define this manifold as the minima of a function that measures the distance between the initial reference image and another generated image. Within this function, the non-mask region of the images are directly compared while the difference between the mask regions of both images is offset by an adjustable parameter, which leads to the function being minimal when the mask region has changed by that factor. 

To create smoothly varying animations that explore this manifold, we implement a "spring" loss with an adjustable rest length term (green connectors in above figure) between vectors to encourage neighbours to be similar but not by too much. Higher-order “stiffener” springs (red connectors) also connect to vectors that are further apart to encourage the entire length of vectors to have minimal curvature. 

## Usage 

First, download the converted StyleGAN2 FFHQ checkpoint for its PyTorch implementation [here](https://drive.google.com/file/d/1v0iLBeuaegDZb3BIBb1CSmfsNSRiNWqI/view?usp=sharing) and put it in the same directory as the rest of the code. 

**Dependencies:**

* PyTorch 1.5
* matplotlib
* numpy
* PIL
* lpips 
* umap 
* tqdm 

### Generate Vectors 

> python run.py 

This method contains different arguments for hyperparameters controlling the weights of each loss term, the mask offset value, and the primary spring loss distance. Additional arguments also specify items such as the mask region, number of output vectors, and optimization epochs. Defaults have been set for all these arguments so the method can be easily used. 

To see a list of all arguments with explanations: 

> python run.py -h 

### Output Files 

Under the default arguments, once the optimization finishes, the resulting latent vectors will be saved in file `{exp_name}_latents.pt`, the corresponding images will be saved in a new directory called `{exp_name}_imgs`, and a plot of the UMAP projection (https://arxiv.org/abs/1802.03426) of the latent vectors will be saved as `{exp_name}_UMAP.png`. `exp_name` is by default set to `masked_gan_exp` but this can be changed by:

> python run.py --exp_name {NEW_EXPERIMENT_NAME}

## Output Examples 

![Boy mouth animation](figures/boy_mouth.gif)

Animation created from images generated using a mask region around the mouth. The mask offset is slowly increased as the animation progresses. 

![UMAP sample](figures/UMAP.png)

A UMAP projection of output latent vectors. 

