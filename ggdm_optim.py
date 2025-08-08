import argparse
from model import MNISTDiffusion
import torch
from utils import ExponentialMovingAverage
from reward_model import ThicknessPredictor
import cv2
import numpy as np
import os

# python ggdm_optim.py --iterations 2 --weights /home/nsortur/GGDMOptim/MNISTDiffusion/results/steps_00046900.pt --rew_model /home/nsortur/GGDMOptim/mnist_thickness/thickness_predictor.pth --target 200 --guidance 2 --n_samples 2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(
        description="Optimization scaffold for GGDMOptim using diffusion weights."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Number of optimization iterations.",
        default=8
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the file containing diffusion weights."
    )
    parser.add_argument(
        "--rew_model",
        type=str,
        required=True,
        help="Path to the weights of the reward model."
    )
    parser.add_argument(
        "--target",
        type=float,
        required=True,
        help="The target reward to achieve."
    )
    parser.add_argument(
        "--guidance",
        type=float,
        required=True,
        help="Guidance strength"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed for the random number generator"
    )
    args = parser.parse_args()
    
    reward_model = ThicknessPredictor().to(device)
    reward_ckpt = torch.load(args.rew_model)
    reward_model.load_state_dict(reward_ckpt)
    
    model = MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=64,
                dim_mults=[2,4]).to(device)

    # these params dont matter for sampling
    ckpt_raw = torch.load(args.weights)['model_ema']
    ckpt = {'.'.join(k.split('.')[1:]): v for k, v in ckpt_raw.items()}
    del ckpt[""]
    model.load_state_dict(ckpt)
    
    model.set_linear_reward_model(is_init=True, batch_size=args.n_samples, height=28, width=28)

    guidances = [args.guidance] * args.iterations
    targets = [args.target] * args.iterations
    
    # use the same random noise so each iteration has the same results
    torch.manual_seed(args.seed)
    image_size = 28
    shape = (args.n_samples, 1, image_size, image_size)
    ### sample the latent variable at t=T from N(0, I) as the start of reverse diffusion model
    init_x = torch.randn(shape, device=device)
    init_x = init_x.to(device)
        
    # afterwards, do for each digit class - right now, only doing one
    opt_images_step = []
    for i in range(args.iterations):
        torch.manual_seed(args.seed)
        model.set_target(targets[i])
        model.set_guidance(guidances[i])
        target_t = torch.tensor([args.target], device=device).repeat(args.n_samples, 1)
        samples_normalized, samples = model.sampling(args.n_samples,clipped_reverse_diffusion=True,device=device, target=target_t, init_x=init_x)
        
        
        ### query reward gradient with regard to the generated images
        grads, biases, rewards = get_grad_eval(samples, reward_model)
        grads = grads.clone().detach()
        biases = biases.clone().detach()    
        model.set_linear_reward_model(gradients = grads, biases = biases, batch_size=args.n_samples, height=28, width=28)
        rewards = rewards.detach().cpu().numpy()
        
        opt_images_step.append(samples_normalized)

    # save the images into opt_results folder
    with torch.no_grad():
        os.makedirs("opt_results", exist_ok=True)
        for i, opt_images in enumerate(opt_images_step):
            for j, img in enumerate(opt_images):
                img = img.cpu().numpy()
                img = img.transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                cv2.imwrite(f"opt_results/opt_images_{i}_{j}.png", img)
            
        
def get_grad_eval(ims, reward_model, device='cuda'):    
    ims = ims.to(device)
    ims.requires_grad = True

    rewards, rewards_std = reward_model(ims)
    print("rewards", rewards)
    
    rewards_squeezed = rewards.squeeze()
    rewards_sum = rewards.sum()
    rewards_sum.backward() # get the gradients for each sample in the batch

    grads = ims.grad
    # print("grads", grads)
    # print("grads shape", grads.shape)
    # print('\n')
    # print("rewards", rewards)
    # print("rewards shape", rewards.shape)
    # print('\n')
    # print("ims", ims)
    # print("ims shape", ims.shape)
    # print('\n')
    biases = - torch.einsum('bijk,bijk->b', grads, ims) + rewards_squeezed # r(x) - <grad, x>
    # print("biases", biases)
    # print("biases shape", biases.shape)
    # print('\n')
    return grads, biases, rewards

if __name__ == "__main__":
    main()