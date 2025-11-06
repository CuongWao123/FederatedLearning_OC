"""Utility functions for logging and experiment management."""

import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from collections.abc import Iterable

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg


# Directory for logs
SERVER_LOG_DIR = Path("server_logs")
SERVER_LOG_DIR.mkdir(exist_ok=True)


def create_experiment_log_file(config: Dict[str, Any]) -> Path:
    """Create experiment log file with header.
    
    Args:
        config: Experiment configuration dict with keys:
            - fraction_train: Client fraction (C)
            - local_epochs: Local epochs (E)
            - batch_size: Batch size (B)
            - lr: Learning rate
            - iid: IID or Non-IID partitioning
            - num_rounds: Max number of rounds
            
    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partition_type = "iid" if config["iid"] else "noniid"
    batch_str = "inf" if config["batch_size"] <= 0 else str(config["batch_size"])
    
    log_filename = (
        f"experiment_C{config['fraction_train']}_"
        f"E{config['local_epochs']}_"
        f"B{batch_str}_"
        f"{partition_type}_"
        f"{timestamp}.txt"
    )
    log_file = SERVER_LOG_DIR / log_filename
    
    batch_size_str = "∞ (full batch)" if config["batch_size"] <= 0 else str(config["batch_size"])
    
    # Write header
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("MNIST 2NN Federated Learning Experiment\n")
        f.write("="*70 + "\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: 2NN (199,210 parameters)\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  Client Fraction (C): {config['fraction_train']}\n")
        f.write(f"  Local Epochs (E): {config['local_epochs']}\n")
        f.write(f"  Batch Size (B): {batch_size_str}\n")
        f.write(f"  Learning Rate: {config['lr']}\n")
        f.write(f"  Data Partitioning: {'IID' if config['iid'] else 'Non-IID'}\n")
        f.write(f"  Max Rounds: {config['num_rounds']}\n")
        f.write("="*70 + "\n\n")
    
    return log_file


def write_experiment_summary(
    log_file: Path,
    train_metrics: Dict[int, Dict[str, float]],
    eval_metrics: Dict[int, Dict[str, float]],
    start_time: datetime,
    end_time: datetime,
    target_accuracy: float = 0.97
) -> None:
    """Write experiment summary to log file.
    
    Args:
        log_file: Path to log file
        train_metrics: Dict of {round_num: {'train_loss': value}}
        eval_metrics: Dict of {round_num: {'eval_acc': value, 'eval_loss': value}}
        start_time: Experiment start time
        end_time: Experiment end time
        target_accuracy: Target accuracy threshold (default: 0.97 = 97%)
    """
    elapsed = end_time - start_time
    
    # Find round where target accuracy was reached
    rounds_to_target = None
    for round_num in sorted(eval_metrics.keys()):
        if eval_metrics[round_num]["eval_acc"] >= target_accuracy:
            rounds_to_target = round_num
            break
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*70 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {elapsed}\n")
        f.write(f"Total Rounds Completed: {len(train_metrics)}\n")
        
        # Target accuracy analysis
        if rounds_to_target:
            f.write(f"\n✓ Target Accuracy ({target_accuracy*100:.0f}%) reached at round: {rounds_to_target}\n")
            final_acc = eval_metrics[rounds_to_target]["eval_acc"]
            f.write(f"  Accuracy at round {rounds_to_target}: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
        else:
            final_round = max(eval_metrics.keys()) if eval_metrics else 0
            if final_round > 0:
                final_acc = eval_metrics[final_round]["eval_acc"]
                f.write(f"\n✗ Target Accuracy ({target_accuracy*100:.0f}%) not reached\n")
                f.write(f"  Best accuracy: {final_acc:.4f} ({final_acc*100:.2f}%) at round {final_round}\n")
        
        # Write all training metrics
        f.write("\n" + "="*70 + "\n")
        f.write("ALL TRAINING METRICS\n")
        f.write("="*70 + "\n")
        for round_num in sorted(train_metrics.keys()):
            metrics = train_metrics[round_num]
            f.write(f"{round_num}: {{'train_loss': '{metrics['train_loss']:.4e}'}}\n")
        
        # Write all evaluation metrics
        f.write("\n" + "="*70 + "\n")
        f.write("ALL EVALUATION METRICS\n")
        f.write("="*70 + "\n")
        for round_num in sorted(eval_metrics.keys()):
            metrics = eval_metrics[round_num]
            f.write(f"{round_num}: {{'eval_acc': '{metrics['eval_acc']:.4e}', 'eval_loss': '{metrics['eval_loss']:.4e}'}}\n")
        
        f.write("\n" + "="*70 + "\n")


class LoggingFedAvg(FedAvg):
    """FedAvg strategy with automatic logging to file.
    
    This strategy extends the standard FedAvg to automatically log
    all outputs to a file instead of console.
    
    Args:
        log_file_path: Path to the log file
        *args: Arguments passed to FedAvg
        **kwargs: Keyword arguments passed to FedAvg
    """
    
    def __init__(self, *args, log_file_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file_path
        self.train_metrics: Dict[int, Dict[str, float]] = {}
        self.eval_metrics: Dict[int, Dict[str, float]] = {}
    
    def _log_to_file(self, message: str) -> None:
        """Write log message to file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    def summary(self) -> None:
        """Log summary configuration to file instead of console."""
        self._log_to_file("\n" + "="*70)
        self._log_to_file("STRATEGY CONFIGURATION")
        self._log_to_file("="*70)
        self._log_to_file("Sampling:")
        self._log_to_file(f"  ├── Fraction: train ({self.fraction_train:.2f}) | evaluate ({self.fraction_evaluate:.2f})")
        self._log_to_file(f"  ├── Minimum nodes: train ({self.min_train_nodes}) | evaluate ({self.min_evaluate_nodes})")
        self._log_to_file(f"  └── Minimum available nodes: {self.min_available_nodes}")
        self._log_to_file("\nKeys in records:")
        self._log_to_file(f"  ├── Weighted by: '{self.weighted_by_key}'")
        self._log_to_file(f"  ├── ArrayRecord key: '{self.arrayrecord_key}'")
        self._log_to_file(f"  └── ConfigRecord key: '{self.configrecord_key}'")
        self._log_to_file("="*70 + "\n")
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure training and log to file."""
        messages = super().configure_train(server_round, arrays, config, grid)
        
        if messages:
            num_nodes = len(list(messages))
            total_nodes = len(list(grid.get_node_ids()))
            self._log_to_file(f"[Round {server_round}] Configure Train: Sampled {num_nodes} nodes (out of {total_nodes})")
        
        return messages
    
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure evaluation and log to file."""
        messages = super().configure_evaluate(server_round, arrays, config, grid)
        
        if messages:
            num_nodes = len(list(messages))
            total_nodes = len(list(grid.get_node_ids()))
            self._log_to_file(f"[Round {server_round}] Configure Evaluate: Sampled {num_nodes} nodes (out of {total_nodes})")
        
        return messages
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate training results and log to file."""
        # Count valid and error replies
        replies_list = list(replies)
        valid_count = sum(1 for msg in replies_list if not msg.has_error())
        error_count = len(replies_list) - valid_count
        
        # Log to file
        self._log_to_file(f"\n[Round {server_round}] Aggregate Train:")
        self._log_to_file(f"  ├── Received {valid_count} results and {error_count} failures")
        
        # Log errors if any
        for msg in replies_list:
            if msg.has_error():
                self._log_to_file(f"  ├── Error from node {msg.metadata.src_node_id}: {msg.error.reason}")
        
        # Call parent aggregate
        arrays, metrics = super().aggregate_train(server_round, replies)
        
        # Extract and store train loss
        if valid_count > 0 and metrics:
            train_losses = []
            for msg in replies_list:
                if not msg.has_error() and "train_loss" in msg.content.get("metrics", {}):
                    train_losses.append(msg.content["metrics"]["train_loss"])
            
            if train_losses:
                avg_loss = sum(train_losses) / len(train_losses)
                self.train_metrics[server_round] = {"train_loss": avg_loss}
                self._log_to_file(f"  └── Average Train Loss: {avg_loss:.4e}")
        
        return arrays, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation results and log to file."""
        # Count valid and error replies
        replies_list = list(replies)
        valid_count = sum(1 for msg in replies_list if not msg.has_error())
        error_count = len(replies_list) - valid_count
        
        # Log to file
        self._log_to_file(f"\n[Round {server_round}] Aggregate Evaluate:")
        self._log_to_file(f"  ├── Received {valid_count} results and {error_count} failures")
        
        # Log errors if any
        for msg in replies_list:
            if msg.has_error():
                self._log_to_file(f"  ├── Error from node {msg.metadata.src_node_id}: {msg.error.reason}")
        
        # Call parent aggregate
        metrics = super().aggregate_evaluate(server_round, replies)
        
        # Extract and store eval metrics
        if valid_count > 0:
            eval_losses = []
            eval_accs = []
            
            for msg in replies_list:
                if not msg.has_error():
                    msg_metrics = msg.content.get("metrics", {})
                    if "eval_loss" in msg_metrics:
                        eval_losses.append(msg_metrics["eval_loss"])
                    if "eval_acc" in msg_metrics:
                        eval_accs.append(msg_metrics["eval_acc"])
            
            if eval_losses and eval_accs:
                avg_loss = sum(eval_losses) / len(eval_losses)
                avg_acc = sum(eval_accs) / len(eval_accs)
                
                self.eval_metrics[server_round] = {
                    "eval_loss": avg_loss,
                    "eval_acc": avg_acc
                }
                
                # Log to file and console (console for quick monitoring)
                status = f"  └── Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f} ({avg_acc*100:.2f}%)"
                self._log_to_file(status)
                print(f"Round {server_round:3d} {status}")  # Also print to console for monitoring
        
        self._log_to_file("-"*70)
        
        return metrics


def save_model(
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    save_dir: Optional[Path] = None
) -> Path:
    """Save model to disk with descriptive filename.
    
    Args:
        state_dict: Model state dictionary
        config: Experiment configuration
        save_dir: Directory to save model (default: current directory)
        
    Returns:
        Path to saved model file
    """
    if save_dir is None:
        save_dir = Path(".")
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    partition_type = "iid" if config["iid"] else "noniid"
    batch_str = "inf" if config["batch_size"] <= 0 else str(config["batch_size"])
    
    filename = (
        f"final_model_2nn_"
        f"C{config['fraction_train']}_"
        f"E{config['local_epochs']}_"
        f"B{batch_str}_"
        f"{partition_type}.pt"
    )
    
    filepath = save_dir / filename
    torch.save(state_dict, filepath)
    
    return filepath


def print_config_summary(config: Dict[str, Any], total_params: int) -> None:
    """Print experiment configuration summary.
    
    Args:
        config: Experiment configuration
        total_params: Total number of model parameters
    """
    batch_size_str = "∞ (full batch)" if config["batch_size"] <= 0 else str(config["batch_size"])
    
    print("=" * 60)
    print("Initializing MNIST 2NN model...")
    print(f"Total parameters: {total_params:,}")
    print(f"\nExperiment Configuration:")
    print(f"  Client Fraction (C): {config['fraction_train']}")
    print(f"  Local Epochs (E): {config['local_epochs']}")
    print(f"  Batch Size (B): {batch_size_str}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Data Partitioning: {'IID' if config['iid'] else 'Non-IID'}")
    print(f"  Max Rounds: {config['num_rounds']}")
    print("=" * 60)