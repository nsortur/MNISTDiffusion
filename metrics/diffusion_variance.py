import sys
sys.path.append("/home/nsortur/GGDMOptim/MNISTDiffusion")

import argparse
from model import MNISTDiffusion
import torch
from utils import ExponentialMovingAverage
from reward_model import ThicknessPredictor
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from torchvision.utils import save_image
import math

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = "/home/nsortur/GGDMOptim/MNISTDiffusion/dilation_results_resume/steps_00046900.pt"
    
    model = MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=64,
                dim_mults=[2,4]).to(device)

    # these params dont matter for sampling
    ckpt_raw = torch.load(weights)['model_ema']
    ckpt = {'.'.join(k.split('.')[1:]): v for k, v in ckpt_raw.items()}
    del ckpt[""]
    model.load_state_dict(ckpt)
    
    model.eval()
    samples_normalized, samples = model.sampling(n_samples=8,clipped_reverse_diffusion=True,device=device,target=None)
    
    # samples, _ = model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
    # save_image(samples,"{}/steps_{:0>8}.png".format(save_dir,global_steps),nrow=int(math.sqrt(args.n_samples)))
        
        
    # save samples as imagesfor visualization
    save_image(samples_normalized, f"/home/nsortur/GGDMOptim/MNISTDiffusion/metrics/diffusion_samples/sample_0.png",nrow=int(math.sqrt(8)))
    # for i in range(len(samples_normalized)):
        
    #     sample = samples_normalized[i].cpu().detach().numpy().squeeze()
    #     sample = (sample + 1) / 2
    #     sample = sample.reshape(28, 28)
    #     plt.imshow(sample, cmap='gray')
    #     # plt.savefig(f"/home/nsortur/GGDMOptim/MNISTDiffusion/metrics/diffusion_samples/sample_{i}.png")
    #     save_image(sample, f"/home/nsortur/GGDMOptim/MNISTDiffusion/metrics/diffusion_samples/sample_{i}.png")
    #     plt.close()

if __name__ == "__main__":
    main()