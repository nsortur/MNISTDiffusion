import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm

class MNISTDiffusion(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8]):
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)
        self.reward_model = None
        self.target_guidance = None
        self.target = None
        self.device = "cuda"
        
    def forward(self,x,noise):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x_t,t)

        return pred_noise
    
    # @torch.no_grad()
    # def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda"):
    #     x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
    #     for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
    #         noise=torch.randn_like(x_t).to(device)
    #         t=torch.tensor([i for _ in range(n_samples)]).to(device)

    #         if clipped_reverse_diffusion:
    #             x_t=self._reverse_diffusion_with_clip(x_t,t,noise)
    #         else:
    #             x_t=self._reverse_diffusion(x_t,t,noise)

    #     x_t=(x_t+1.)/2. #[-1,1] to [0,1]

    #     return x_t

    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda", target=None, init_x=None):
        if init_x is not None:
            x_t=init_x
        else:
            x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
            
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            with torch.enable_grad():
                x_t.requires_grad_(True)
                noise=torch.randn_like(x_t).to(device)
                
                t=torch.tensor([i for _ in range(n_samples)]).to(device)

                if clipped_reverse_diffusion:
                    x_t=self._reverse_diffusion_with_clip(x_t,t,noise,target=target, device=device, batch_size=n_samples)
                else:
                    x_t=self._reverse_diffusion(x_t,t,noise,target=target, device=device, batch_size=n_samples)
                
                # print(x_t)
                # if i == 997:
                #     return x_t
            
            x_t = x_t.detach()

        x_t_normalized=(x_t+1.)/2. #[-1,1] to [0,1]
        # x_t = x_t+0.0

        return x_t_normalized, x_t
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    def _reverse_diffusion(self,x_t,t,noise,target=None):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        x_t.requires_grad_(True)
        pred=self.model(x_t,t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0
            
        raise ValueError()
        # if target is not None:
        #     sqrt_alpha_t = torch.sqrt(alpha_t)

        #     clean_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_t
        #     out, _ = self.reward_model(clean_pred)
        #     print(out)
            
        #     l2_error = 0.5 * torch.nn.MSELoss()(out, target)
        #     gradient_guidance = torch.autograd.grad(l2_error, x_t)[0]
            
        #     # add guidance to noise
        #     noise += sqrt_one_minus_alpha_cumprod_t * self.target_guidance * gradient_guidance

        return mean+std*noise 
    
    # @torch.no_grad()
    # def _reverse_diffusion_with_clip(self,x_t,t,noise): 
    #     '''
    #     p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

    #     pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
    #     '''
    #     pred=self.model(x_t,t)
    #     alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
    #     beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
    #     x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
    #     x_0_pred.clamp_(-1., 1.)

    #     if t.min()>0:
    #         alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
    #         mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
    #              ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

    #         std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
    #     else:
    #         mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
    #         std=0.0

    #     return mean+std*noise 


    def _reverse_diffusion_with_clip(self,x_t,t,noise,device,batch_size,target=None): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)
        # step_match = t[0].item() >= self.timesteps - 40
        step_match = ((t[0].item()+1) % 100 == 0) or (t[0].item() == 0)
        # if step_match:
        #     print("pred", pred)
        #     print("pred shape", pred.shape)
        #     print('\n')

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_t_cumprod)

        # x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred=(x_t-torch.sqrt(1. - alpha_t_cumprod)*pred) / sqrt_alpha_cumprod_t
        x_0_pred.clamp_(-1., 1.)
        
        
        if target is not None:
            target = torch.FloatTensor([[self.target]]).to(device)
            target = target.repeat(batch_size, 1)
            
            # sqrt_alpha_t = torch.sqrt(alpha_t_cumprod)

            # make sure we're doing linear approximation of reward
            # clean_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_t
            # if step_match:
            #     print("clean pred", clean_pred)
            #     print("clean pred shape", clean_pred.shape)
            #     print('\n')
                
            # grad shape torch.Size([4, 3, 32, 32])
            # Clean pred torch.Size([4, 1, 32, 32])
            # Bias shape torch.Size([4, 4])

            out = torch.einsum('bijk,bijk->b', self.gradients, x_0_pred) + self.biases
            # if step_match:
                # print("out", out)
                # print("out shape", out.shape)
                # print('\n')
            
            # print(out)
            # print(out.shape) # torch.Size([4, 4])
            # print('\n')
            # print(target)
            # print(target.shape) # torch.Size([4, 1])
            # print('\n')
            # out, _ = self.reward_model(clean_pred)
            # print(out)
            
            l2_error = 0.5 * torch.nn.MSELoss()(out, target)
            if step_match:
                print("l2 error", l2_error)
                print("target", target)
                print("out", out)
                print('\n')
            gradient_guidance = torch.autograd.grad(l2_error, x_t)[0]
            # if step_match:
            #     print("gradient guidance", gradient_guidance)
            #     print("gradient guidance shape", gradient_guidance.shape)
            #     print('\n')
            
            # add guidance to noise
            pred += sqrt_one_minus_alpha_cumprod_t * self.target_guidance * gradient_guidance
        
        
        x_0_pred=(x_t-torch.sqrt(1. - alpha_t_cumprod)*pred) / sqrt_alpha_cumprod_t
        x_0_pred.clamp_(-1., 1.)
        
        # This works:
        # python ggdm_optim.py --iterations 2 --weights /home/nsortur/GGDMOptim/MNISTDiffusion/dilation_results/steps_00046900.pt --rew_model /home/nsortur/GGDMOptim/mnist_thickness/thickness_predictor.pth --target 100 --guidance 0.01 --n_samples 4 --seed 123456
        
        # copy from grad guided sdpipeline
        # x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * pred) / sqrt_alpha_cumprod_t
        path = "/home/nsortur/GGDMOptim/MNISTDiffusion/x0_preds_debug/"
        # if t[0].item() <= self.timesteps - 980:
        if step_match:
            import os
            import torchvision.utils as vutils
            os.makedirs(path, exist_ok=True)
            # Clamp to [-1, 1], then scale to [0, 1] for saving
            x0_img = x_0_pred.clone().detach().cpu()
            x0_img = (x0_img + 1) / 2
            fname = f"x0_pred_t{t[0].item()}.png"
            vutils.save_image(x0_img, os.path.join(path, fname))
        
        # if t[0].item() == 0:
        #     import sys
        #     sys.exit(0)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0
        

        return mean+std*noise
    
    def set_guidance(self, guidance):
        self.target_guidance = guidance 
        
    # def set_reward_model(self, model):
    #     self.reward_model = model
    
    def set_linear_reward_model(self, gradients = None, biases = None, is_init = False, batch_size = 1, height = 28, width = 28):
        if is_init:
            self.gradients = torch.zeros(batch_size, 3, height, width)
            self.biases = torch.zeros(1)
        else:
            self.gradients = gradients
            self.biases = biases
        
        self.gradients.requires_grad_(False)
        self.biases.requires_grad_(False)
        self.gradients = self.gradients.to(self.device)
        self.biases = self.biases.to(self.device)
        
        # print("gradients", self.gradients)
        # print("biases", self.biases)
        # print("biases shape", self.biases.shape)
        # print('\n')
        
    def set_target(self, target):
        self.target = target