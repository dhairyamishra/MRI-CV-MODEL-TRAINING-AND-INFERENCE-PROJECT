#!/usr/bin/env python3
"""
Real-time Training Monitor
===========================

Monitor training progress in real-time with live plots and metrics.

Usage:
    # Monitor classification training
    python scripts/monitor_training.py --task classification
    
    # Monitor segmentation training
    python scripts/monitor_training.py --task segmentation
    
    # Monitor with auto-refresh
    python scripts/monitor_training.py --task classification --refresh 5
"""

import sys
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TrainingMonitor:
    """Real-time training monitor with live plots."""
    
    def __init__(self, task, log_dir=None, refresh_interval=5):
        self.task = task
        self.refresh_interval = refresh_interval
        
        # Determine log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        elif task == "classification":
            self.log_dir = project_root / "logs" / "classification_production"
        elif task == "segmentation":
            self.log_dir = project_root / "logs" / "seg_production"
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Initialize data storage
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_metric = []
        self.val_metric = []
        self.learning_rates = []
        
        # Metric names
        self.metric_name = "ROC-AUC" if task == "classification" else "Dice"
        
        # Setup plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup matplotlib figure with subplots."""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Loss curves
        self.ax_loss = self.fig.add_subplot(gs[0, :])
        self.ax_loss.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Metric curves
        self.ax_metric = self.fig.add_subplot(gs[1, :])
        self.ax_metric.set_title(f'Training & Validation {self.metric_name}', fontsize=14, fontweight='bold')
        self.ax_metric.set_xlabel('Epoch')
        self.ax_metric.set_ylabel(self.metric_name)
        self.ax_metric.grid(True, alpha=0.3)
        
        # Learning rate
        self.ax_lr = self.fig.add_subplot(gs[2, 0])
        self.ax_lr.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.set_yscale('log')
        self.ax_lr.grid(True, alpha=0.3)
        
        # Statistics
        self.ax_stats = self.fig.add_subplot(gs[2, 1])
        self.ax_stats.axis('off')
        
        plt.suptitle(f'SliceWise {self.task.capitalize()} Training Monitor', 
                     fontsize=16, fontweight='bold')
        
    def parse_log_file(self):
        """Parse training log file to extract metrics."""
        # Look for latest log file
        log_files = list(self.log_dir.glob("*.log"))
        if not log_files:
            return False
        
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            # Parse metrics from log
            # This is a simplified parser - adjust based on actual log format
            for line in lines:
                if 'Epoch' in line and 'Loss' in line:
                    # Extract epoch, loss, and metrics
                    # Format example: "Epoch 10/100 - Train Loss: 0.123, Val Loss: 0.456, Val Dice: 0.789"
                    pass
            
            return True
        except Exception as e:
            print(f"Error parsing log: {e}")
            return False
    
    def parse_wandb_logs(self):
        """Parse W&B logs if available."""
        # Check for wandb run directory
        wandb_dir = project_root / "wandb"
        if not wandb_dir.exists():
            return False
        
        # Find latest run
        run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith('run-')]
        if not run_dirs:
            return False
        
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        
        # Try to parse wandb history
        history_file = latest_run / "files" / "wandb-history.jsonl"
        if not history_file.exists():
            return False
        
        try:
            self.epochs = []
            self.train_loss = []
            self.val_loss = []
            self.train_metric = []
            self.val_metric = []
            self.learning_rates = []
            
            with open(history_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    
                    # Extract epoch
                    if 'epoch' in data:
                        epoch = data['epoch']
                        if epoch not in self.epochs:
                            self.epochs.append(epoch)
                    
                    # Extract losses
                    if 'train_loss' in data:
                        self.train_loss.append(data['train_loss'])
                    if 'val_loss' in data:
                        self.val_loss.append(data['val_loss'])
                    
                    # Extract metrics
                    if self.task == "classification":
                        if 'train_roc_auc' in data:
                            self.train_metric.append(data['train_roc_auc'])
                        if 'val_roc_auc' in data:
                            self.val_metric.append(data['val_roc_auc'])
                    else:  # segmentation
                        if 'train_dice' in data:
                            self.train_metric.append(data['train_dice'])
                        if 'val_dice' in data:
                            self.val_metric.append(data['val_dice'])
                    
                    # Extract learning rate
                    if 'learning_rate' in data:
                        self.learning_rates.append(data['learning_rate'])
            
            return True
        except Exception as e:
            print(f"Error parsing W&B logs: {e}")
            return False
    
    def parse_checkpoint_metadata(self):
        """Parse checkpoint metadata for metrics."""
        checkpoint_dir = project_root / "checkpoints"
        if self.task == "classification":
            checkpoint_dir = checkpoint_dir / "cls_production"
        else:
            checkpoint_dir = checkpoint_dir / "seg_production"
        
        if not checkpoint_dir.exists():
            return False
        
        # Look for metrics.json or similar
        metrics_file = checkpoint_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Extract data
                if 'epochs' in data:
                    self.epochs = data['epochs']
                if 'train_loss' in data:
                    self.train_loss = data['train_loss']
                if 'val_loss' in data:
                    self.val_loss = data['val_loss']
                
                return True
            except Exception as e:
                print(f"Error parsing metrics: {e}")
                return False
        
        return False
    
    def update_plot(self, frame=None):
        """Update plot with latest data."""
        # Try to load data from various sources
        success = (self.parse_wandb_logs() or 
                  self.parse_checkpoint_metadata() or 
                  self.parse_log_file())
        
        if not success or not self.epochs:
            return
        
        # Clear axes
        self.ax_loss.clear()
        self.ax_metric.clear()
        self.ax_lr.clear()
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Plot loss
        if self.train_loss:
            self.ax_loss.plot(self.epochs[:len(self.train_loss)], self.train_loss, 
                            'b-', label='Train Loss', linewidth=2)
        if self.val_loss:
            self.ax_loss.plot(self.epochs[:len(self.val_loss)], self.val_loss, 
                            'r-', label='Val Loss', linewidth=2)
        self.ax_loss.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        # Plot metric
        if self.train_metric:
            self.ax_metric.plot(self.epochs[:len(self.train_metric)], self.train_metric, 
                              'b-', label=f'Train {self.metric_name}', linewidth=2)
        if self.val_metric:
            self.ax_metric.plot(self.epochs[:len(self.val_metric)], self.val_metric, 
                              'r-', label=f'Val {self.metric_name}', linewidth=2)
        self.ax_metric.set_title(f'Training & Validation {self.metric_name}', 
                                fontsize=14, fontweight='bold')
        self.ax_metric.set_xlabel('Epoch')
        self.ax_metric.set_ylabel(self.metric_name)
        self.ax_metric.legend()
        self.ax_metric.grid(True, alpha=0.3)
        
        # Plot learning rate
        if self.learning_rates:
            self.ax_lr.plot(self.epochs[:len(self.learning_rates)], self.learning_rates, 
                          'g-', linewidth=2)
        self.ax_lr.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.set_yscale('log')
        self.ax_lr.grid(True, alpha=0.3)
        
        # Display statistics
        stats_text = self.generate_stats_text()
        self.ax_stats.text(0.1, 0.9, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=11, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    def generate_stats_text(self):
        """Generate statistics text."""
        if not self.epochs:
            return "No data available"
        
        current_epoch = self.epochs[-1] if self.epochs else 0
        
        stats = f"Training Statistics\n"
        stats += f"{'='*40}\n"
        stats += f"Current Epoch: {current_epoch}\n"
        stats += f"Last Updated: {datetime.now().strftime('%H:%M:%S')}\n\n"
        
        if self.train_loss:
            stats += f"Train Loss: {self.train_loss[-1]:.4f}\n"
        if self.val_loss:
            stats += f"Val Loss:   {self.val_loss[-1]:.4f}\n"
            if len(self.val_loss) > 1:
                best_val_loss = min(self.val_loss)
                stats += f"Best Val Loss: {best_val_loss:.4f}\n"
        
        stats += f"\n"
        
        if self.train_metric:
            stats += f"Train {self.metric_name}: {self.train_metric[-1]:.4f}\n"
        if self.val_metric:
            stats += f"Val {self.metric_name}:   {self.val_metric[-1]:.4f}\n"
            if len(self.val_metric) > 1:
                best_val_metric = max(self.val_metric)
                stats += f"Best Val {self.metric_name}: {best_val_metric:.4f}\n"
        
        if self.learning_rates:
            stats += f"\nLearning Rate: {self.learning_rates[-1]:.2e}\n"
        
        return stats
    
    def run(self, live=True):
        """Run the monitor."""
        if live:
            print(f"Starting live training monitor for {self.task}...")
            print(f"Monitoring directory: {self.log_dir}")
            print(f"Refresh interval: {self.refresh_interval}s")
            print("Press Ctrl+C to stop\n")
            
            # Setup animation
            ani = animation.FuncAnimation(
                self.fig, 
                self.update_plot, 
                interval=self.refresh_interval * 1000,
                cache_frame_data=False
            )
            
            plt.show()
        else:
            # Single update
            self.update_plot()
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time training monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation"],
        required=True,
        help="Which task to monitor"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom log directory to monitor"
    )
    
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval in seconds"
    )
    
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live updates (single snapshot)"
    )
    
    args = parser.parse_args()
    
    try:
        monitor = TrainingMonitor(
            task=args.task,
            log_dir=args.log_dir,
            refresh_interval=args.refresh
        )
        monitor.run(live=not args.no_live)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
