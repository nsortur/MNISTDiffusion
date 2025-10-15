import torch
import sys
sys.path.append(".")

from unet import Unet

def debug_unet_channels():
    """Debug the channel dimensions in UNet"""
    
    print("=== Debugging UNet Channel Dimensions ===\n")
    
    # Test with the working config first
    print("Testing dim_mults = [2, 4]")
    model = Unet(
        timesteps=1000,
        time_embedding_dim=128,
        in_channels=1,
        out_channels=1,
        base_dim=64,
        dim_mults=[2, 4]
    )
    
    print(f"Base dim: 64")
    print(f"Channels: {model._cal_channels(64, [2, 4])}")
    
    # Print encoder and decoder block info
    print(f"\nEncoder blocks:")
    for i, block in enumerate(model.encoder_blocks):
        print(f"  Block {i}: {block}")
    
    print(f"\nDecoder blocks:")
    for i, block in enumerate(model.decoder_blocks):
        print(f"  Block {i}: {block}")
    
    print(f"\nMid block: {model.mid_block}")
    print(f"Final conv: {model.final_conv}")
    
    # Test forward pass
    try:
        x = torch.randn(2, 1, 28, 28)
        t = torch.randint(0, 1000, (2,))
        y = model(x, t)
        print(f"\n✓ Forward pass successful! Output shape: {y.shape}")
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_unet_channels() 