import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
args = parser.parse_args()


def adjust_scab_influence(model, epoch, total_epochs, warmup_epochs=10):
    """
    更激进的SCAB权重调度，让SCAB从一开始就充分参与
    
    调度策略：
    - Epoch 0-10: 从0.7增长到0.9 (高起点，让SCAB立即发挥作用)
    - Epoch 10+: 从0.9增长到1.0 (逐渐释放全部能力)
    
    对比旧方案：
    - 旧：0.5 -> 0.8 -> 1.0
    - 新：0.7 -> 0.9 -> 1.0
    提高了20%的初始权重
    """
    if not hasattr(model, 'scab_modules') or model.scab_modules is None:
        return 0.0
    
    if epoch <= warmup_epochs:
        # ⚠️ 关键修改：从0.7开始而不是0.5
        # 原理：配合SCAB内部0.989的输出，总体特征保留率 = 0.7 × 0.989 ≈ 0.69
        target_weight = 0.7 + (0.9 - 0.7) * (epoch / warmup_epochs)  # 0.7 -> 0.9
    else:
        # 后续从0.9缓慢增长到1.0
        remaining_epochs = total_epochs - warmup_epochs
        progress = min(1.0, (epoch - warmup_epochs) / remaining_epochs)
        target_weight = 0.9 + 0.1 * progress  # 0.9 -> 1.0
    
    # 更新所有SCAB模块的权重
    for scab_module in model.scab_modules:
        if hasattr(scab_module, 'attention_weight'):
            with torch.no_grad():
                scab_module.attention_weight.data.fill_(target_weight)
    
    return target_weight


def load_pretrained_weights_selective(model, pretrained_path, ignore_scab=True, logger=None):
    """
    Load pretrained weights while ignoring SCAB-related parameters
    """
    if not os.path.exists(pretrained_path):
        if logger:
            logger.warning(f"Pretrained weights not found at {pretrained_path}")
        return model
    
    if logger:
        logger.info(f"Loading pretrained weights from {pretrained_path}")
    
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
        if logger:
            logger.info(f"Filtered out {len(pretrained_dict) - len(filtered_dict)} SCAB-related parameters")
    else:
        filtered_dict = pretrained_dict
    
    # Update only existing parameters
    matched_dict = {k: v for k, v in filtered_dict.items() if k in model_dict}
    missing_keys = set(model_dict.keys()) - set(matched_dict.keys())
    unexpected_keys = set(filtered_dict.keys()) - set(model_dict.keys())
    
    if logger:
        logger.info(f"Matched parameters: {len(matched_dict)}")
        logger.info(f"Missing parameters: {len(missing_keys)}")
        logger.info(f"Unexpected parameters: {len(unexpected_keys)}")
    
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    
    return model


class ProgressiveSCABTrainer(Trainer):
    """
    Extended trainer with SCAB progressive training
    """
    def __init__(self, cfg, model, optimizer, train_loader, test_loader, lr_scheduler, warmup_lr_scheduler, logger, loss, model_name):
        # Initialize parent class normally (handles pretrained weight loading with flexible logic)
        super().__init__(cfg, model, optimizer, train_loader, test_loader, lr_scheduler, warmup_lr_scheduler, logger, loss, model_name)
        self.scab_warmup_epochs = cfg.get('scab_warmup_epochs', 20)
        
        # Log SCAB status
        if hasattr(model, 'scab_modules') and model.scab_modules is not None:
            self.logger.info(f"SCAB modules initialized: {len(model.scab_modules)} modules")
        else:
            self.logger.info("No SCAB modules found")
        
    def train(self):
        """Override train method to add SCAB progress tracking"""
        start_epoch = self.epoch
        
        # Log SCAB configuration
        if hasattr(self.model, 'scab_modules') and self.model.scab_modules is not None:
            self.logger.info(f"SCAB Progressive Training Configuration:")
            self.logger.info(f"  - SCAB modules: {len(self.model.scab_modules)}")
            self.logger.info(f"  - Warmup epochs: {self.scab_warmup_epochs}")
            self.logger.info(f"  - Total epochs: {self.cfg['max_epoch']}")
        
        # Call parent train method
        super().train()
        
    def train_one_epoch(self, epoch):
        # Adjust SCAB influence before training epoch
        if hasattr(self.model, 'scab_modules') and self.model.scab_modules is not None:
            scab_weight = adjust_scab_influence(
                self.model, epoch, self.cfg['max_epoch'], self.scab_warmup_epochs
            )
            if epoch % 10 == 0 or epoch <= self.scab_warmup_epochs:  # Log every 10 epochs or during warmup
                self.logger.info(f"Epoch {epoch}: SCAB weight = {scab_weight:.4f}")
        
        # Call parent training method
        return super().train_one_epoch(epoch)


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    logger.info("=" * 50)
    logger.info("Progressive SCAB Training Started")
    logger.info("=" * 50)
    logger.info(f"Config: {args.config}")
    logger.info(f"SCAB enabled: {cfg['model'].get('use_scab', False)}")
    logger.info(f"SCAB reduction: {cfg['model'].get('scab_reduction', 16)}")

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return
    #ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    # Check if SCAB is enabled and choose appropriate trainer
    use_scab = cfg['model'].get('use_scab', False)
    
    if use_scab:
        logger.info("Using Progressive SCAB Trainer")
        initial_scab_weight = adjust_scab_influence(model, 0, cfg['trainer']['max_epoch'])
        logger.info(f"Initial SCAB weight: {initial_scab_weight:.4f}")
        
        # Build Progressive SCAB trainer
        trainer = ProgressiveSCABTrainer(cfg=cfg['trainer'],
                          model=model,
                          optimizer=optimizer,
                          train_loader=train_loader,
                          test_loader=test_loader,
                          lr_scheduler=lr_scheduler,
                          warmup_lr_scheduler=warmup_lr_scheduler,
                          logger=logger,
                          loss=loss,
                          model_name=model_name)
    else:
        logger.info("Using Standard Trainer")
        # Build standard trainer
        trainer = Trainer(cfg=cfg['trainer'],
                          model=model,
                          optimizer=optimizer,
                          train_loader=train_loader,
                          test_loader=test_loader,
                          lr_scheduler=lr_scheduler,
                          warmup_lr_scheduler=warmup_lr_scheduler,
                          logger=logger,
                          loss=loss,
                          model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()


if __name__ == '__main__':
    main()
