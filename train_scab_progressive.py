#!/usr/bin/env python3
"""
Progressive training script for SCAB integration
This script gradually increases the influence of SCAB modules during training
"""

import torch
import torch.nn as nn
import yaml
import argparse
import os
import sys
import numpy as np

# Add project path
sys.path.append('/home/zjhzjh/workdir/Modify_MonoDGP')

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger

def adjust_scab_influence(model, epoch, total_epochs, warmup_epochs=20):
    """
    Gradually increase SCAB influence during training
    """
    if not hasattr(model, 'scab_modules') or model.scab_modules is None:
        return
    
    if epoch <= warmup_epochs:
        # Gradual warmup: 0.01 -> 0.5
        target_weight = 0.01 + (0.5 - 0.01) * (epoch / warmup_epochs)
    else:
        # After warmup: 0.5 -> 1.0
        remaining_epochs = total_epochs - warmup_epochs
        progress = min(1.0, (epoch - warmup_epochs) / remaining_epochs)
        target_weight = 0.5 + 0.5 * progress
    
    # Update all SCAB modules
    for scab_module in model.scab_modules:
        if hasattr(scab_module, 'attention_weight'):
            with torch.no_grad():
                scab_module.attention_weight.data.fill_(target_weight)
    
    return target_weight

def load_pretrained_weights_selective(model, pretrained_path, ignore_scab=True):
    """
    Load pretrained weights while ignoring SCAB-related parameters
    """
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        return model
    
    print(f"Loading pretrained weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint
    
    model_dict = model.state_dict()
    
    # Filter out SCAB-related parameters if requested
    if ignore_scab:
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if 'scab_modules' not in k}
        print(f"Filtered out {len(pretrained_dict) - len(filtered_dict)} SCAB-related parameters")
    else:
        filtered_dict = pretrained_dict
    
    # Update only existing parameters
    matched_dict = {k: v for k, v in filtered_dict.items() if k in model_dict}
    missing_keys = set(model_dict.keys()) - set(matched_dict.keys())
    unexpected_keys = set(filtered_dict.keys()) - set(model_dict.keys())
    
    print(f"Matched parameters: {len(matched_dict)}")
    print(f"Missing parameters: {len(missing_keys)}")
    print(f"Unexpected parameters: {len(unexpected_keys)}")
    
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    
    return model

class ProgressiveSCABTrainer(Trainer):
    """
    Extended trainer with SCAB progressive training
    """
    def __init__(self, cfg, model, optimizer, train_loader, test_loader, lr_scheduler, logger):
        super().__init__(cfg, model, optimizer, train_loader, test_loader, lr_scheduler, logger)
        self.scab_warmup_epochs = cfg.get('scab_warmup_epochs', 20)
        
    def train_one_epoch(self, epoch):
        # Adjust SCAB influence
        scab_weight = adjust_scab_influence(
            self.model, epoch, self.max_epoch, self.scab_warmup_epochs
        )
        
        self.logger.info(f"Epoch {epoch}: SCAB weight = {scab_weight:.4f}")
        
        # Call parent training method
        return super().train_one_epoch(epoch)

def main():
    parser = argparse.ArgumentParser(description='Progressive SCAB Training')
    parser.add_argument('--config', default='configs/monodgp_scab.yaml', 
                       help='Path to config file')
    parser.add_argument('--pretrained', default=None,
                       help='Path to pretrained weights')
    parser.add_argument('--gpu', default='0', help='GPU id to use')
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create logger
    logger = create_logger(cfg['trainer']['save_path'])
    logger.info("=" * 50)
    logger.info("Progressive SCAB Training Started")
    logger.info("=" * 50)
    logger.info(f"Config: {args.config}")
    logger.info(f"SCAB enabled: {cfg['model'].get('use_scab', False)}")
    logger.info(f"SCAB reduction: {cfg['model'].get('scab_reduction', 16)}")
    
    # Build components
    logger.info("Building model...")
    model, criterion = build_model(cfg['model'])
    
    # Load pretrained weights (excluding SCAB parameters)
    if args.pretrained:
        model = load_pretrained_weights_selective(model, args.pretrained, ignore_scab=True)
    elif 'pretrain_model' in cfg['trainer']:
        model = load_pretrained_weights_selective(
            model, cfg['trainer']['pretrain_model'], ignore_scab=True
        )
    
    logger.info("Building data loaders...")
    train_loader = build_dataloader(cfg['dataset'], 'train')
    test_loader = build_dataloader(cfg['dataset'], 'test')
    
    logger.info("Building optimizer and scheduler...")
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer)
    
    # Initialize SCAB weights to very small values
    initial_scab_weight = adjust_scab_influence(model, 0, cfg['trainer']['max_epoch'])
    logger.info(f"Initial SCAB weight: {initial_scab_weight:.4f}")
    
    # Build trainer
    trainer = ProgressiveSCABTrainer(
        cfg['trainer'], model, optimizer, train_loader, test_loader, lr_scheduler, logger
    )
    
    # Start training
    logger.info("Starting progressive training...")
    trainer.train()
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 