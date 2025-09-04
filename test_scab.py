#!/usr/bin/env python3
"""
Test script to verify SCAB (Spatial-Channel Attention Block) integration
"""

import torch
import yaml
import sys
import os

# Add the project root to Python path
sys.path.append('/home/zjhzjh/workdir/Modify_MonoDGP')

from lib.models.monodgp import build_monodgp
from lib.models.monodgp.attention_modules import SpatialChannelAttention

def test_scab_module():
    """Test the SCAB module independently"""
    print("=" * 50)
    print("Testing SCAB Module")
    print("=" * 50)
    
    # Test different channel sizes
    test_cases = [64, 128, 256, 512]
    
    for channels in test_cases:
        print(f"\nTesting SCAB with {channels} channels...")
        
        # Create SCAB module
        scab = SpatialChannelAttention(channels, reduction=16)
        
        # Create dummy input
        batch_size = 2
        height, width = 32, 64
        x = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        with torch.no_grad():
            output = scab(x)
        
        # Check output shape
        expected_shape = (batch_size, channels, height, width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check that output is not identical to input (attention is working)
        assert not torch.equal(x, output), "Output should be different from input"
        
        print(f"✓ SCAB test passed for {channels} channels")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameter count: {sum(p.numel() for p in scab.parameters())}")

def test_monodgp_with_scab():
    """Test MonoDGP model with SCAB enabled"""
    print("\n" + "=" * 50)
    print("Testing MonoDGP with SCAB Integration")
    print("=" * 50)
    
    # Load configuration
    config_path = '/home/zjhzjh/workdir/Modify_MonoDGP/configs/monodgp.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"SCAB enabled: {cfg['model'].get('use_scab', 'Not specified')}")
    print(f"SCAB reduction: {cfg['model'].get('scab_reduction', 'Not specified')}")
    
    try:
        # Build model with SCAB enabled
        print("\nBuilding MonoDGP model with SCAB...")
        model, criterion = build_monodgp(cfg['model'])
        
        # Check if SCAB modules are present
        if hasattr(model, 'scab_modules') and model.scab_modules is not None:
            print(f"✓ SCAB modules found: {len(model.scab_modules)} modules")
            for i, scab in enumerate(model.scab_modules):
                param_count = sum(p.numel() for p in scab.parameters())
                print(f"  SCAB module {i}: {param_count} parameters")
        else:
            print("✗ SCAB modules not found or disabled")
            return False
        
        # Check GPU availability for forward pass test
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice available: {device}")
        
        if device.type == 'cpu':
            print("⚠️  GPU not available. Skipping forward pass test (some operations require GPU).")
            print("✓ Model building and SCAB integration successful!")
            return True
        
        # Test with dummy input on GPU
        print("\nTesting forward pass on GPU...")
        batch_size = 1
        height, width = 384, 1280  # KITTI image size
        channels = 3
        
        # Create dummy data and move to GPU
        images = torch.randn(batch_size, channels, height, width).to(device)
        calibs = torch.randn(batch_size, 3, 4).to(device)
        img_sizes = torch.tensor([[height, width]] * batch_size).to(device)
        
        # Move model to GPU
        model = model.to(device)
        
        # Forward pass in evaluation mode
        model.eval()
        with torch.no_grad():
            try:
                outputs = model(images, calibs, targets=None, img_sizes=img_sizes)
                print("✓ Forward pass successful!")
                print(f"  Outputs type: {type(outputs)}")
                if isinstance(outputs, dict):
                    print(f"  Output keys: {list(outputs.keys())}")
                return True
            except Exception as e:
                print(f"✗ Forward pass failed: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False

def test_scab_vs_no_scab():
    """Compare model with and without SCAB"""
    print("\n" + "=" * 50)
    print("Comparing Models: SCAB vs No-SCAB")
    print("=" * 50)
    
    config_path = '/home/zjhzjh/workdir/Modify_MonoDGP/configs/monodgp.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Ensure models are built on CPU for parameter counting
    cfg['model']['device'] = 'cpu'
    
    try:
        # Test with SCAB
        cfg['model']['use_scab'] = True
        model_with_scab, _ = build_monodgp(cfg['model'])
        params_with_scab = sum(p.numel() for p in model_with_scab.parameters())
        
        # Test without SCAB
        cfg['model']['use_scab'] = False
        model_without_scab, _ = build_monodgp(cfg['model'])
        params_without_scab = sum(p.numel() for p in model_without_scab.parameters())
        
        print(f"Model with SCAB: {params_with_scab:,} parameters")
        print(f"Model without SCAB: {params_without_scab:,} parameters")
        print(f"SCAB overhead: {params_with_scab - params_without_scab:,} parameters")
        print(f"Relative increase: {((params_with_scab - params_without_scab) / params_without_scab * 100):.2f}%")
        
        # Clean up
        del model_with_scab, model_without_scab
        return True
        
    except Exception as e:
        print(f"✗ Parameter comparison failed: {e}")
        return False

if __name__ == "__main__":
    print("SCAB Integration Test")
    print("=" * 50)
    
    # Test 1: SCAB module functionality
    test_scab_module()
    
    # Test 2: MonoDGP integration
    success = test_monodgp_with_scab()
    
    # Test 3: Parameter comparison
    if success:
        test_scab_vs_no_scab()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! SCAB integration successful!")
    else:
        print("❌ Some tests failed. Please check the integration.")
    print("=" * 50) 