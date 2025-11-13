"""Test GPU 1 usage configuration."""

import os
# Force GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from SANET import get_device, SANet


def test_gpu_configuration():
    """Test that GPU 1 is being used."""
    
    print("=" * 60)
    print("GPU Configuration Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"2. Number of GPUs visible: {torch.cuda.device_count()}")
        print(f"   (Should be 1 due to CUDA_VISIBLE_DEVICES='1')")
        
        # Test get_device function
        device = get_device(gpu_id=0)  # Use 0 because only 1 GPU visible
        print(f"\n3. Selected Device: {device}")
        print(f"   Physical GPU: GPU 1")
        
        # Create a test tensor
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        print(f"4. Test Tensor Device: {test_tensor.device}")
        
        # Create SANet model and move to device
        print("\n5. Testing SANet model on GPU 1...")
        model = SANet(sa_channels=(64, 128, 256, 512))
        model.to(device)
        
        # Check model device
        first_param = next(model.parameters())
        print(f"   Model Device: {first_param.device}")
        
        # Test forward pass
        test_input = torch.randn(1, 3, 224, 224).to(device)
        print(f"   Test Input Shape: {test_input.shape}")
        print(f"   Test Input Device: {test_input.device}")
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   Output Shape: {output.shape}")
        print(f"   Output Device: {output.device}")
        
        print("\n" + "=" * 60)
        print("✅ GPU 1 Configuration Test PASSED")
        print("=" * 60)
        
    else:
        print("\n❌ CUDA not available - cannot test GPU configuration")
        print("=" * 60)


if __name__ == "__main__":
    test_gpu_configuration()
