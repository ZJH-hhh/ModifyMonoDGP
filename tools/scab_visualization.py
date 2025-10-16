import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_scab_logs(log_path):
    """Load SCAB scheduler logs from JSON file"""
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_weight_evolution(logs, save_path=None):
    """Plot SCAB weight evolution over training"""
    weight_history = logs['weight_history']
    
    if not weight_history:
        print("No weight history found in logs")
        return
    
    epochs = [w['epoch'] for w in weight_history]
    base_weights = [w['base_weight'] for w in weight_history]
    adaptive_mults = [w['adaptive_mult'] for w in weight_history]
    avg_weights = [w['avg_weight'] for w in weight_history]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Base weight evolution
    ax1.plot(epochs, base_weights, 'b-', linewidth=2, label='Base Weight')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Base Weight')
    ax1.set_title('SCAB Base Weight Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Adaptive multiplier
    ax2.plot(epochs, adaptive_mults, 'r-', linewidth=2, label='Adaptive Multiplier')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Adaptive Multiplier')
    ax2.set_title('Adaptive Adjustment Multiplier')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Average final weight
    ax3.plot(epochs, avg_weights, 'g-', linewidth=2, label='Average Final Weight')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Weight')
    ax3.set_title('Final SCAB Weight (Layer Average)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Layer-wise weights (if available)
    if weight_history and 'layer_weights' in weight_history[0]:
        num_layers = len(weight_history[0]['layer_weights'])
        for layer_idx in range(num_layers):
            layer_weights = [w['layer_weights'][layer_idx] for w in weight_history]
            ax4.plot(epochs, layer_weights, linewidth=1.5, 
                    label=f'Layer {layer_idx}', alpha=0.8)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Layer Weight')
        ax4.set_title('SCAB Weights by Layer')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight evolution plot saved to {save_path}")
    else:
        plt.show()


def plot_performance_trends(logs, save_path=None):
    """Plot performance trends and adaptive adjustments"""
    adjustment_history = logs.get('adjustment_history', [])
    loss_history = logs.get('loss_history', [])
    ap_history = logs.get('ap_history', [])
    
    if not adjustment_history:
        print("No adjustment history found in logs")
        return
    
    epochs = [adj['epoch'] for adj in adjustment_history]
    loss_trends = [adj['loss_trend'] for adj in adjustment_history]
    ap_trends = [adj['ap_trend'] for adj in adjustment_history]
    adjustments = [adj['adjustment'] for adj in adjustment_history]
    stagnations = [adj['stagnation'] for adj in adjustment_history]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss trend
    ax1.plot(epochs, loss_trends, 'r-', linewidth=2, label='Loss Trend')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No Change')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Trend (slope)')
    ax1.set_title('Loss Trend Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # AP trend
    ax2.plot(epochs, ap_trends, 'g-', linewidth=2, label='AP Trend')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No Change')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AP Trend (slope)')
    ax2.set_title('AP Trend Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adaptive adjustments
    ax3.plot(epochs, adjustments, 'b-', linewidth=2, label='Adjustment Factor')
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No Adjustment')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Adjustment Factor')
    ax3.set_title('Adaptive Adjustment Factor')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Stagnation counter
    ax4.plot(epochs, stagnations, 'orange', linewidth=2, label='Stagnation Counter')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Stagnation Count')
    ax4.set_title('Performance Stagnation Detection')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance trends plot saved to {save_path}")
    else:
        plt.show()


def plot_training_metrics(logs, save_path=None):
    """Plot raw training metrics if available"""
    loss_history = logs.get('loss_history', [])
    ap_history = logs.get('ap_history', [])
    
    if not loss_history and not ap_history:
        print("No training metrics found in logs")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss history
    if loss_history:
        ax1.plot(range(len(loss_history)), loss_history, 'r-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # AP history
    if ap_history:
        ax2.plot(range(len(ap_history)), ap_history, 'g-', linewidth=2, label='3D AP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('3D AP')
        ax2.set_title('3D AP Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {save_path}")
    else:
        plt.show()


def generate_analysis_report(logs, output_path=None):
    """Generate a comprehensive analysis report"""
    stats = logs.get('statistics', {})
    weight_history = logs.get('weight_history', [])
    adjustment_history = logs.get('adjustment_history', [])
    
    report = []
    report.append("=" * 60)
    report.append("ENHANCED SCAB TRAINING ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # General statistics
    if stats:
        report.append("GENERAL STATISTICS:")
        report.append(f"  Total weight updates: {stats.get('total_updates', 0)}")
        report.append(f"  Current average weight: {stats.get('current_avg_weight', 0):.4f}")
        report.append(f"  Weight variance: {stats.get('weight_variance', 0):.6f}")
        report.append(f"  Strategy used: {stats.get('strategy', 'unknown')}")
        report.append(f"  Layer-wise scheduling: {stats.get('layer_wise', False)}")
        report.append(f"  Adaptive adjustment: {stats.get('adaptive', False)}")
        report.append(f"  Adaptive adjustments made: {stats.get('adaptive_adjustments', 0)}")
        report.append(f"  Max stagnation periods: {stats.get('stagnation_periods', 0)}")
        report.append("")
    
    # Weight evolution analysis
    if weight_history:
        final_weights = weight_history[-1]['layer_weights'] if 'layer_weights' in weight_history[-1] else []
        report.append("WEIGHT EVOLUTION ANALYSIS:")
        report.append(f"  Initial average weight: {weight_history[0]['avg_weight']:.4f}")
        report.append(f"  Final average weight: {weight_history[-1]['avg_weight']:.4f}")
        report.append(f"  Weight growth: {weight_history[-1]['avg_weight'] - weight_history[0]['avg_weight']:.4f}")
        
        if final_weights:
            report.append(f"  Final layer weights: {[f'{w:.3f}' for w in final_weights]}")
            report.append(f"  Layer weight std: {np.std(final_weights):.4f}")
        report.append("")
    
    # Adaptive behavior analysis
    if adjustment_history:
        adjustments = [adj['adjustment'] for adj in adjustment_history]
        non_identity_adjustments = [adj for adj in adjustments if abs(adj - 1.0) > 0.01]
        
        report.append("ADAPTIVE BEHAVIOR ANALYSIS:")
        report.append(f"  Total adaptive events: {len(non_identity_adjustments)}")
        if non_identity_adjustments:
            report.append(f"  Average adjustment factor: {np.mean(adjustments):.4f}")
            report.append(f"  Min adjustment factor: {np.min(adjustments):.4f}")
            report.append(f"  Max adjustment factor: {np.max(adjustments):.4f}")
        
        stagnation_events = [adj for adj in adjustment_history if adj['stagnation'] > 0]
        if stagnation_events:
            report.append(f"  Stagnation events: {len(stagnation_events)}")
            report.append(f"  Max stagnation duration: {max(adj['stagnation'] for adj in stagnation_events)}")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    if stats.get('adaptive_adjustments', 0) == 0:
        report.append("  - Consider enabling adaptive adjustment for better responsiveness")
    elif stats.get('adaptive_adjustments', 0) > 50:
        report.append("  - High number of adjustments - consider reducing sensitivity")
    
    if stats.get('weight_variance', 0) > 0.01:
        report.append("  - High weight variance across layers - verify layer-wise scheduling")
    
    if stats.get('stagnation_periods', 0) > 10:
        report.append("  - Frequent stagnation detected - consider adjusting learning rate or strategy")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Analysis report saved to {output_path}")
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='SCAB Training Visualization and Analysis')
    parser.add_argument('log_path', help='Path to SCAB scheduler logs JSON file')
    parser.add_argument('--output_dir', default='./scab_analysis', help='Output directory for plots and reports')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load logs
    try:
        logs = load_scab_logs(args.log_path)
        print(f"Loaded SCAB logs from {args.log_path}")
    except Exception as e:
        print(f"Error loading logs: {e}")
        return
    
    # Generate plots
    plot_weight_evolution(logs, 
                         save_path=None if args.show_plots else output_dir / 'weight_evolution.png')
    
    plot_performance_trends(logs, 
                           save_path=None if args.show_plots else output_dir / 'performance_trends.png')
    
    plot_training_metrics(logs, 
                         save_path=None if args.show_plots else output_dir / 'training_metrics.png')
    
    # Generate analysis report
    generate_analysis_report(logs, output_path=output_dir / 'analysis_report.txt')
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main() 