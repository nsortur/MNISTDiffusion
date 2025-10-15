import torch
import sys
sys.path.append(".")

from unet import Unet

def test_unet_configs():
    """Test UNet creation with different dim_mults configurations"""
    
    # Test configurations
    configs = [
        [2, 4],      # Original working config
        [1, 2, 4, 8], # Config that was failing
        [1, 2, 4],   # Another config
        [2, 4, 8],   # Another config
    ]
    
    for dim_mults in configs:
        try:
            print(f"Testing dim_mults = {dim_mults}")
            model = Unet(
                timesteps=1000,
                time_embedding_dim=128,
                in_channels=1,
                out_channels=1,
                base_dim=64,
                dim_mults=dim_mults
            )
            
            # Test forward pass
            x = torch.randn(2, 1, 28, 28)
            t = torch.randint(0, 1000, (2,))
            y = model(x, t)
            print(f"  ✓ Success! Output shape: {y.shape}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        print()

if __name__ == "__main__":
    test_unet_configs() 