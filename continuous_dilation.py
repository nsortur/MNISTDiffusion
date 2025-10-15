import torch
import torchvision.transforms as T
import kornia.morphology as km
import random

# Define a custom PyTorch-like transform class
class RandomGrayscaleDilation(object):
    def __init__(self, kernel_bounds=(1, 4), seed=None):
        self.kernel_bounds = kernel_bounds
        self.seed = seed
        # Create a separate random state for this transform instance
        if self.seed is not None:
            self.rng = random.Random(self.seed)
        else:
            self.rng = random.Random()

    def __call__(self, img):
        # Add a batch dimension if the image doesn't have one (C, H, W -> 1, C, H, W)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        # Apply the grayscale dilation. Kornia's dilation function expects a tensor
        # of shape (B, C, H, W).
        kernel_size = self.rng.randint(self.kernel_bounds[0], self.kernel_bounds[1] + 1)
        kernel = torch.ones(kernel_size, kernel_size)
        dilated_img = km.dilation(img, kernel.to(img.device))
        
        # Remove the batch dimension if it was added
        return dilated_img.squeeze(0)

    def set_seed(self, seed):
        """Update the seed for this transform instance"""
        self.seed = seed
        if self.seed is not None:
            self.rng = random.Random(self.seed)
        else:
            self.rng = random.Random()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.datasets import MNIST

    # Load MNIST dataset (just for demonstration)
    transform = T.Compose([
        T.ToTensor(),  # Convert PIL Image to PyTorch tensor
        T.Resize((28, 28)),
        # GrayscaleDilation(kernel=torch.ones(8, 8)),
        # Add other transforms here if needed, e.g., T.Normalize
        # T.Normalize([0.5], [0.5])
    ])
    mnist = MNIST(root="./mnist_data", train=True, download=True, transform=transform)

    # Get a sample image and label
    img, label = mnist[1]

    # Apply the grayscale dilation transform
    dilation_transform = RandomGrayscaleDilation(kernel_bounds=(1, 4))
    dilated_img = dilation_transform(img)
    
    print(dilated_img)

    # Print the label and show original and dilated images
    print(f"Label: {label}")
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze().numpy(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dilated_img.squeeze().numpy(), cmap='gray')
    plt.title("Dilated")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("dilation.png")
    # # Now you can use this transform in your pipeline
    