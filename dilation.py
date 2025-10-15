import torch
import torch.nn.functional as F
import numpy as np
import random
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from torchvision import transforms


class RandomBinaryDilation:
    """
    A custom torch transform that applies random binary dilation to images.
    
    Args:
        dilation_prob (float): Probability of applying dilation (default: 0.7)
        max_dilation_iterations (int): Maximum number of dilation iterations (default: 3)
        threshold (float): Threshold for converting to binary image (default: 0.5)
    """
    
    def __init__(self, dilation_prob=0.7, max_dilation_iterations=3, threshold=0.5):
        self.dilation_prob = dilation_prob
        self.max_dilation_iterations = max_dilation_iterations
        self.threshold = threshold
    
    def __call__(self, image):
        """
        Apply random binary dilation to the input image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W) or (H, W)
            
        Returns:
            torch.Tensor: Dilated image tensor with same shape as input
        """
        # Apply dilation with probability
        if random.random() < self.dilation_prob:
            # Handle different input shapes
            if image.dim() == 3:  # (C, H, W)
                # For multi-channel images, apply dilation to each channel
                dilated_channels = []
                for c in range(image.shape[0]):
                    channel = image[c]
                    dilated_channel = self._apply_dilation_to_channel(channel)
                    dilated_channels.append(dilated_channel)
                return torch.stack(dilated_channels)
            elif image.dim() == 2:  # (H, W)
                return self._apply_dilation_to_channel(image)
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got {image.dim()}D")
        
        return image
    
    def _apply_dilation_to_channel(self, channel):
        """
        Apply binary dilation to a single channel.
        
        Args:
            channel (torch.Tensor): Single channel tensor of shape (H, W)
            
        Returns:
            torch.Tensor: Dilated channel tensor
        """
        # Convert to numpy for scipy operations
        channel_np = channel.numpy()
        
        # Convert to binary image using threshold
        binary_image = (channel_np > self.threshold).astype(np.uint8)
        
        # Random number of dilation iterations
        iterations = random.randint(1, self.max_dilation_iterations)
        
        # Create structuring element (3x3 cross)
        structure = generate_binary_structure(2, 1)
        
        # Apply binary dilation
        dilated_image = binary_image
        for _ in range(iterations):
            dilated_image = binary_dilation(dilated_image, structure=structure)
        
        # Convert back to tensor and normalize to [0, 1]
        dilated_tensor = torch.from_numpy(dilated_image.astype(np.float32))
        
        return dilated_tensor
    
    
# Example usage:
if __name__ == "__main__":
    # Create the transform
    dilation_transform = RandomBinaryDilation(
        dilation_prob=0.7,
        max_dilation_iterations=3,
        threshold=0.5
    )
    
    # Example: Use in a transform pipeline
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor [0, 1]
        dilation_transform,     # Apply random binary dilation
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Example: Apply to MNIST dataset
    from torchvision.datasets import MNIST
    
    # Create dataset with dilation transform
    mnist_dataset = MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=transform_pipeline
    )
    
    # Test the transform
    import matplotlib.pyplot as plt
    
    # Get a sample image
    sample_image, label = mnist_dataset[0]
    
    # Visualize original vs dilated
    plt.figure(figsize=(10, 5))
    
    # Original image (without dilation)
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    original_dataset = MNIST(root="./mnist_data", train=True, download=True, transform=original_transform)
    original_image, _ = original_dataset[0]
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title('Original MNIST Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title('MNIST Image with Random Dilation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dilation_demo.png')
    
    print("Transform created successfully!")
    print(f"Dilation probability: {dilation_transform.dilation_prob}")
    print(f"Max dilation iterations: {dilation_transform.max_dilation_iterations}")
    print(f"Binary threshold: {dilation_transform.threshold}")