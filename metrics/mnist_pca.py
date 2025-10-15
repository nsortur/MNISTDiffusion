# perform PCA on MNIST, use that as A
# Use the on-off subspace method from Appendix E.1.1 to evaluate

# doesn't really work though because PCA for A is just linear and not great

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.decomposition import PCA
from dataset import create_mnist_dataloaders
import numpy as np
import pickle

def on_subspace(pca, data):
    A = pca.components_.T
    return A @ A.T @ data

def off_subspace(pca, data):
    A = pca.components_.T
    I = torch.eye(A.shape[0])
    return (I - A @ A.T) @ data
    
    
def ground_truth_pca(pca, train_data):
    samp_test = train_data[0]
    on_subspace_test = on_subspace(pca, samp_test)
    off_subspace_test = off_subspace(pca, samp_test)
    # magnitude of on and off subspace
    off_subspace_test_norm = np.linalg.norm(off_subspace_test)
    on_subspace_test_norm = np.linalg.norm(on_subspace_test)
    print("Ratio of off and on subspace: ", off_subspace_test_norm / on_subspace_test_norm)
    print("Norm of off subspace: ", off_subspace_test_norm)
    print("Norm of on subspace: ", on_subspace_test_norm)
    
    assert np.isclose(off_subspace_test_norm**2 + on_subspace_test_norm**2, np.linalg.norm(samp_test)**2, atol=1e-6), \
        f"off_subspace_test_norm**2 + on_subspace_test_norm**2 != np.linalg.norm(samp_test)**2: {off_subspace_test_norm**2 + on_subspace_test_norm**2} != {np.linalg.norm(samp_test)**2}"
        
        
def diffusion_pca(pca):
    # load the generated data
    with open("/home/nsortur/GGDMOptim/MNISTDiffusion/metrics/generated_data/samples_unnormalized_step.pkl", "rb") as f:
        samples_unnormalized_step = pickle.load(f)
    
    for i in range(len(samples_unnormalized_step)):
        quotients = []
        for j in range(len(samples_unnormalized_step[i])):
            sample = samples_unnormalized_step[i][j].cpu().detach().numpy().flatten()
            on_subspace_sample = on_subspace(pca, sample)
            off_subspace_sample = off_subspace(pca, sample)
            quotient = np.linalg.norm(off_subspace_sample) / np.linalg.norm(on_subspace_sample)
            quotients.append(quotient)
        quotients = np.array(quotients)
        print(f"Step {i}: mean={quotients.mean():.6f}, std={quotients.std():.6f}")

def main():
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=10000, dilation=True)
    train_data = train_dataloader.dataset.data
    train_data = train_data.view(train_data.shape[0], -1).numpy()
    
    pca = PCA(0.98)
    pca.fit(train_data)
    
    # ground_truth_pca(pca, train_data)
    diffusion_pca(pca)

if __name__ == "__main__":
    main()