import torch
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class EnhancedSCABScheduler:
    """
    Enhanced SCAB weight scheduler with multiple strategies and adaptive mechanisms
    """
    
    def __init__(self, 
                 total_epochs: int = 250,
                 warmup_epochs: int = 20,
                 strategy: str = 'cosine_restart',
                 layer_wise: bool = True,
                 adaptive: bool = True,
                 monitor_window: int = 5):
        """
        Args:
            total_epochs: Total training epochs
            warmup_epochs: Warmup period for SCAB
            strategy: Scheduling strategy ('linear', 'cosine', 'exponential', 'cosine_restart')
            layer_wise: Whether to use different schedules for different layers
            adaptive: Whether to use adaptive adjustment based on training state
            monitor_window: Window size for performance monitoring
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy
        self.layer_wise = layer_wise
        self.adaptive = adaptive
        self.monitor_window = monitor_window
        
        # Performance tracking
        self.loss_history = []
        self.ap_history = []
        self.weight_history = []
        self.adjustment_history = []
        
        # Adaptive parameters
        self.base_learning_rate = 1.0
        self.patience = 5
        self.stagnation_counter = 0
        self.last_best_epoch = 0
        self.last_best_ap = 0.0
        
        print(f"Enhanced SCAB Scheduler initialized:")
        print(f"  Strategy: {strategy}")
        print(f"  Layer-wise scheduling: {layer_wise}")
        print(f"  Adaptive adjustment: {adaptive}")
    
    def get_base_weight(self, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
        """Get base weight using different scheduling strategies"""
        
        if epoch <= warmup_epochs:
            # Warmup phase: gentle start
            progress = epoch / warmup_epochs
            
            if self.strategy == 'linear':
                return 0.01 + (0.3 - 0.01) * progress
            elif self.strategy == 'cosine':
                return 0.01 + (0.3 - 0.01) * (1 - math.cos(progress * math.pi)) / 2
            elif self.strategy == 'exponential':
                return 0.01 * (30 ** progress)  # 0.01 -> 0.3
            else:  # cosine_restart
                return 0.01 + (0.3 - 0.01) * (1 - math.cos(progress * math.pi)) / 2
        
        else:
            # Main training phase
            remaining_epochs = total_epochs - warmup_epochs
            progress = (epoch - warmup_epochs) / remaining_epochs
            
            if self.strategy == 'linear':
                return 0.3 + (1.0 - 0.3) * progress
            elif self.strategy == 'cosine':
                return 0.3 + (1.0 - 0.3) * (1 - math.cos(progress * math.pi)) / 2
            elif self.strategy == 'exponential':
                return 0.3 + (1.0 - 0.3) * (progress ** 2)
            elif self.strategy == 'cosine_restart':
                # Cosine annealing with restarts
                cycle_length = remaining_epochs // 3  # 3 cycles
                current_cycle = int(progress * 3)
                cycle_progress = (progress * 3) % 1
                
                base_weight = 0.3 + (1.0 - 0.3) * current_cycle / 3
                cycle_weight = base_weight + 0.2 * (1 - math.cos(cycle_progress * math.pi)) / 2
                return min(1.0, cycle_weight)
    
    def get_layer_multiplier(self, layer_idx: int, num_layers: int) -> float:
        """Get layer-specific multiplier for differentiated scheduling"""
        if not self.layer_wise:
            return 1.0
        
        # Earlier layers (lower resolution) get higher weights initially
        # Later layers (higher resolution) get more gradual introduction
        if layer_idx < num_layers // 2:
            return 1.2  # Earlier introduction for low-res features
        else:
            return 0.8  # Later introduction for high-res features
    
    def get_adaptive_adjustment(self, epoch: int, current_loss: float, current_ap: float) -> float:
        """Calculate adaptive adjustment based on training state"""
        if not self.adaptive or epoch < self.monitor_window:
            return 1.0
        
        # Update history
        self.loss_history.append(current_loss)
        self.ap_history.append(current_ap)
        
        # Keep only recent history
        if len(self.loss_history) > self.monitor_window * 2:
            self.loss_history = self.loss_history[-self.monitor_window * 2:]
            self.ap_history = self.ap_history[-self.monitor_window * 2:]
        
        # Calculate trends
        recent_losses = self.loss_history[-self.monitor_window:]
        recent_aps = self.ap_history[-self.monitor_window:]
        
        if len(recent_losses) < self.monitor_window:
            return 1.0
        
        # Loss trend (negative slope means decreasing loss - good)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # AP trend (positive slope means increasing AP - good)
        ap_trend = np.polyfit(range(len(recent_aps)), recent_aps, 1)[0]
        
        # Performance improvement
        if current_ap > self.last_best_ap:
            self.last_best_ap = current_ap
            self.last_best_epoch = epoch
            self.stagnation_counter = 0
            adjustment = 1.0  # Normal progression
        else:
            self.stagnation_counter += 1
            
            if self.stagnation_counter > self.patience:
                # Performance stagnation - slow down SCAB introduction
                adjustment = 0.9
            else:
                adjustment = 1.0
        
        # Trend-based adjustment
        if loss_trend > 0:  # Loss increasing - problematic
            adjustment *= 0.95
        if ap_trend < 0:    # AP decreasing - problematic  
            adjustment *= 0.95
        
        self.adjustment_history.append({
            'epoch': epoch,
            'loss_trend': loss_trend,
            'ap_trend': ap_trend,
            'adjustment': adjustment,
            'stagnation': self.stagnation_counter
        })
        
        return adjustment
    
    def update_scab_weights(self, 
                          model, 
                          epoch: int, 
                          current_loss: float = None, 
                          current_ap: float = None) -> Dict:
        """
        Update SCAB weights with enhanced scheduling
        
        Returns:
            Dict with weight information and statistics
        """
        if not hasattr(model, 'scab_modules') or model.scab_modules is None:
            return {'status': 'no_scab_modules'}
        
        num_layers = len(model.scab_modules)
        layer_weights = []
        
        # Get base weight from scheduling strategy
        base_weight = self.get_base_weight(epoch, self.total_epochs, self.warmup_epochs)
        
        # Get adaptive adjustment
        adaptive_mult = 1.0
        if current_loss is not None and current_ap is not None:
            adaptive_mult = self.get_adaptive_adjustment(epoch, current_loss, current_ap)
        
        # Apply to each layer
        for layer_idx, scab_module in enumerate(model.scab_modules):
            # Layer-specific multiplier
            layer_mult = self.get_layer_multiplier(layer_idx, num_layers)
            
            # Calculate final weight
            final_weight = base_weight * layer_mult * adaptive_mult
            final_weight = max(0.01, min(1.0, final_weight))  # Clamp to valid range
            
            # Update weight
            if hasattr(scab_module, 'attention_weight'):
                with torch.no_grad():
                    scab_module.attention_weight.data.fill_(final_weight)
            
            layer_weights.append(final_weight)
        
        # Record weight history
        self.weight_history.append({
            'epoch': epoch,
            'base_weight': base_weight,
            'adaptive_mult': adaptive_mult,
            'layer_weights': layer_weights.copy(),
            'avg_weight': np.mean(layer_weights)
        })
        
        return {
            'status': 'updated',
            'base_weight': base_weight,
            'adaptive_multiplier': adaptive_mult,
            'layer_weights': layer_weights,
            'avg_weight': np.mean(layer_weights),
            'strategy': self.strategy,
            'stagnation_counter': self.stagnation_counter
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about weight scheduling"""
        if not self.weight_history:
            return {'status': 'no_history'}
        
        recent_weights = self.weight_history[-10:] if len(self.weight_history) >= 10 else self.weight_history
        
        stats = {
            'total_updates': len(self.weight_history),
            'current_avg_weight': recent_weights[-1]['avg_weight'] if recent_weights else 0,
            'weight_variance': np.var([w['avg_weight'] for w in recent_weights]) if len(recent_weights) > 1 else 0,
            'adaptive_adjustments': len([adj for adj in self.adjustment_history if adj['adjustment'] != 1.0]),
            'stagnation_periods': max(self.stagnation_counter, max([adj['stagnation'] for adj in self.adjustment_history], default=0)),
            'strategy': self.strategy,
            'layer_wise': self.layer_wise,
            'adaptive': self.adaptive
        }
        
        return stats
    
    def save_logs(self, save_path: str):
        """Save scheduling logs for analysis"""
        import json
        logs = {
            'weight_history': self.weight_history,
            'adjustment_history': self.adjustment_history,
            'loss_history': self.loss_history,
            'ap_history': self.ap_history,
            'statistics': self.get_statistics()
        }
        
        with open(save_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"SCAB scheduler logs saved to {save_path}") 