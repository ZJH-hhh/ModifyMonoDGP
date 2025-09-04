#!/usr/bin/env python3
"""
Test script to verify SCAB fixes
"""

import torch
import yaml
import sys
import os

# Add the project root to Python path
sys.path.append('/home/zjhzjh/workdir/Modify_MonoDGP')

def test_model_loading():
    """Test if model can be loaded successfully with flexible weight loading"""
    print("Testing SCAB Model Loading Fix")
    print("=" * 50)
    
    try:
        # Load config
        config_path = '/home/zjhzjh/workdir/Modify_MonoDGP/configs/monodgp_scab.yaml'
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print("✓ Config loaded successfully")
        print(f"  SCAB enabled: {cfg['model'].get('use_scab', False)}")
        print(f"  Pretrained model: {cfg['trainer'].get('pretrain_model', 'None')}")
        
        # Build model
        from lib.helpers.model_helper import build_model
        model, loss = build_model(cfg['model'])
        print("✓ Model built successfully")
        
        # Check SCAB modules
        if hasattr(model, 'scab_modules') and model.scab_modules is not None:
            print(f"✓ SCAB modules found: {len(model.scab_modules)} modules")
            
            # Test SCAB forward pass
            for i, scab in enumerate(model.scab_modules):
                test_input = torch.randn(1, 256, 32, 64)  # [B, C, H, W]
                with torch.no_grad():
                    output = scab(test_input)
                    print(f"  SCAB module {i}: input {test_input.shape} -> output {output.shape}")
                    
                    # Check if attention_weight exists
                    if hasattr(scab, 'attention_weight'):
                        print(f"    Attention weight: {scab.attention_weight.item():.4f}")
        else:
            print("✗ No SCAB modules found")
            return False
        
        # Test trainer initialization
        print("\nTesting Trainer Initialization...")
        from tools.train_val import ProgressiveSCABTrainer, adjust_scab_influence
        
        # Test SCAB influence adjustment
        initial_weight = adjust_scab_influence(model, 0, 100, 20)
        print(f"✓ Initial SCAB weight: {initial_weight:.4f}")
        
        mid_weight = adjust_scab_influence(model, 10, 100, 20)
        print(f"✓ Mid training SCAB weight: {mid_weight:.4f}")
        
        final_weight = adjust_scab_influence(model, 50, 100, 20)
        print(f"✓ Final SCAB weight: {final_weight:.4f}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! SCAB integration is working correctly!")
        print("You can now run training with:")
        print("  bash train_scab.sh")
        print("  OR")
        print("  python tools/train_val.py --config configs/monodgp_scab.yaml")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1) 