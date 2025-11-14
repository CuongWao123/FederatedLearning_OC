"""SANet Crowd Counting: Flower Server App for Federated Learning."""

import torch
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.common import EvaluateRes, Scalar

# Import từ SANET.py ở thư mục gốc
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from SANET import SANet

# Create ServerApp
app = ServerApp()


class SANetFedAvg(FedAvg):
    """
    Custom FedAvg strategy for SANet with best model tracking.
    
    Tracks and saves:
    - Best MAE (Mean Absolute Error)
    - Best RMSE (Root Mean Square Error)
    - Training history
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mae = float('inf')
        self.best_rmse = float('inf')
        self.best_round = 0
        self.best_arrays = None
        self.history = {
            'rounds': [],
            'train_loss': [],
            'eval_loss': [],
            'mae': [],
            'rmse': []
        }
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,  # Remove type hint to be flexible
        failures=None,  # Make it optional with default
    ):
        """Aggregate evaluation metrics and track best model."""
        
        if not results:
            return None, {}
        
        # Call parent aggregation - handle both signatures
        if failures is not None:
            parent_result = super().aggregate_evaluate(server_round, results, failures)
        else:
            parent_result = super().aggregate_evaluate(server_round, results)
        
        # Unpack result
        if isinstance(parent_result, tuple):
            loss_aggregated, metrics_aggregated = parent_result
        else:
            loss_aggregated = None
            metrics_aggregated = parent_result if parent_result else {}
        
        # Aggregate MAE and RMSE (weighted by number of examples)
        total_examples = 0
        mae_sum = 0.0
        rmse_sum = 0.0
        
        for item in results:
            # Handle both (num_examples, res) and other formats
            if isinstance(item, tuple) and len(item) == 2:
                num_examples, res = item
                # Get metrics from res
                if hasattr(res, 'metrics'):
                    metrics = res.metrics
                else:
                    metrics = res if isinstance(res, dict) else {}
                
                mae = metrics.get("mae", 0.0)
                rmse = metrics.get("rmse", 0.0)
                
                total_examples += num_examples
                mae_sum += mae * num_examples
                rmse_sum += rmse * num_examples
        
        if total_examples > 0:
            mae_weighted = mae_sum / total_examples
            rmse_weighted = rmse_sum / total_examples
            
            # Add to metrics
            if metrics_aggregated is None:
                metrics_aggregated = {}
            metrics_aggregated["mae"] = mae_weighted
            metrics_aggregated["rmse"] = rmse_weighted
            
            # Track history
            self.history['rounds'].append(server_round)
            self.history['eval_loss'].append(loss_aggregated if loss_aggregated else 0.0)
            self.history['mae'].append(mae_weighted)
            self.history['rmse'].append(rmse_weighted)
            
            # Check if this is the best model
            improved = False
            if mae_weighted < self.best_mae:
                self.best_mae = mae_weighted
                self.best_rmse = rmse_weighted
                self.best_round = server_round
                improved = True
                print(f"\n{'🎯'*35}")
                print(f"NEW BEST MODEL at Round {server_round}!")
                print(f"  MAE:  {mae_weighted:.2f} persons (↓ improved)")
                print(f"  RMSE: {rmse_weighted:.2f} persons")
                print(f"{'🎯'*35}\n")
            
            # Print current metrics
            if not improved:
                print(f"\n{'='*70}")
                print(f"[Round {server_round:3d}] Evaluation Results:")
                if loss_aggregated is not None:
                    print(f"  Loss: {loss_aggregated:.6f}")
                print(f"  MAE:  {mae_weighted:.2f} persons")
                print(f"  RMSE: {rmse_weighted:.2f} persons")
                print(f"  Best MAE so far: {self.best_mae:.2f} (Round {self.best_round})")
                print(f"{'='*70}\n")
        
        return loss_aggregated, metrics_aggregated

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for SANet Federated Learning Server.
    
    Paper: Scale Aggregation Network for Accurate and Efficient Crowd Counting (ECCV 2018)
    
    Configuration:
    - num-server-rounds: Number of federated rounds (paper: 400 epochs)
    - fraction-train: Fraction of clients to sample per round
    - lr: Learning rate (paper: 1e-5)
    - local-epochs: Local training epochs per round
    - batch-size: Batch size (default: 1 due to variable patch sizes)
    - csv-path: Path to dataset CSV file
    """
    
    # Parse configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config.get("fraction-train", 0.1)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    lr = context.run_config.get("lr", 1e-5)
    local_epochs = context.run_config.get("local-epochs", 1)
    csv_path = context.run_config["csv-path"]
    batch_size = context.run_config.get("batch-size", 1)
    partitioner_type = context.run_config.get("partitioner-type", "iid")
    
    # Print configuration
    print("\n" + "="*70)
    print("SANet Federated Crowd Counting")
    print("Paper: Scale Aggregation Network (ECCV 2018)")
    print("="*70)
    print("\n📋 Configuration:")
    print(f"  Server rounds:      {num_rounds}")
    print(f"  Fraction train:     {fraction_train}")
    print(f"  Fraction evaluate:  {fraction_evaluate}")
    print(f"  Learning rate:      {lr}")
    print(f"  Local epochs:       {local_epochs}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Partitioner:        {partitioner_type}")
    print(f"  CSV path:           {csv_path}")
    print("="*70 + "\n")

    # Initialize global SANet model with PAPER CONFIG
    print("🔧 Initializing SANet model...")
    sa_channels = (64, 128, 256, 512)  # ✅ CHUẨN PAPER
    global_model = SANet(sa_channels=sa_channels)
    
    # Count parameters by module FIRST
    print(f"\n{'='*70}")
    print("SANet Parameters Grouped by Module")
    print(f"{'='*70}")
    
    module_params = {}
    total_params = 0
    trainable_params = 0
    
    for name, param in global_model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        # Group by module (sa1, sa2, conv9, etc.)
        module_name = name.split('.')[0]
        if module_name not in module_params:
            module_params[module_name] = 0
        module_params[module_name] += num_params
    
    # Print module summary FIRST
    print(f"\n{'Module':<25} {'Parameters':>15} {'Percentage':>12}")
    print(f"{'-'*70}")
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    for module, count in sorted_modules:
        percentage = (count / total_params) * 100
        print(f"{module:<25} {count:>15,} {percentage:>11.2f}%")
    
    print(f"\n{'-'*70}")
    print(f"{'TOTAL PARAMETERS':<25} {total_params:>15,}")
    print(f"{'Trainable Parameters':<25} {trainable_params:>15,}")
    print(f"{'Non-trainable':<25} {(total_params - trainable_params):>15,}")
    
    # Model size
    param_size_fp32 = total_params * 4 / 1024**2
    param_size_fp16 = total_params * 2 / 1024**2
    
    print(f"\n{'Model Size (FP32):':<25} {param_size_fp32:>10.2f} MB")
    print(f"{'Model Size (FP16/AMP):':<25} {param_size_fp16:>10.2f} MB")
    print(f"{'='*70}")
    
    # OPTIONAL: Detailed layer-by-layer (chỉ in nếu debug mode)
    if context.run_config.get("debug-params", False):
        print(f"\n{'='*70}")
        print("Detailed Layer-by-Layer Parameters")
        print(f"{'='*70}")
        print(f"\n{'Layer Name':<45} {'Parameters':>12} {'Trainable':>8}")
        print(f"{'-'*70}")
        
        for name, param in global_model.named_parameters():
            num_params = param.numel()
            trainable_str = "✓" if param.requires_grad else "✗"
            print(f"{name:<45} {num_params:>12,} {trainable_str:>8}")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<45} {total_params:>12,}")
        print(f"{'='*70}")
    
    print()  # Empty line
    
    # Create initial parameter arrays
    arrays = ArrayRecord(global_model.state_dict())

    # Prepare training configuration
    train_config = ConfigRecord({
        "lr": lr,
        "local-epochs": local_epochs,
        "csv-path": csv_path,
        "batch-size": batch_size,
        "sigma": context.run_config.get("sigma", 4),
        "partitioner-type": partitioner_type,
        "dirichlet-alpha": context.run_config.get("dirichlet-alpha", 0.5),
        "amp": context.run_config.get("amp", True),
        "grad-clip": context.run_config.get("grad-clip", 0.0),
        "alpha": context.run_config.get("alpha", 1e-3),  # SSIM loss weight
        "beta": context.run_config.get("beta", 1e-3),    # Count loss weight
        "num-workers": context.run_config.get("num-workers", 4),
        "pin-memory": context.run_config.get("pin-memory", True),
        "sa-channels": list(sa_channels),  # ✅ Convert tuple to list
    })

    # Initialize custom FedAvg strategy with best model tracking
    print("📊 Initializing SANet FedAvg strategy with best model tracking...")
    print(f"  Fraction fit:       {fraction_train}")
    print(f"  Fraction evaluate:  {fraction_evaluate}\n")
    
    strategy = FedAvg(fraction_train=fraction_train)

    # Start federated learning
    print("🚀 Starting Federated Learning...\n")
    start_time = datetime.now()
    
    # Use strategy to orchestrate federated learning
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds
    )
    
    final_arrays = result.arrays
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\n⏰ Federated Learning completed in: {elapsed}\n")
    #Create models directory
    print("\n" + "="*70)
    print("💾 Saving models and training history...")
    print("\n" + "="*70)
    print("💾 Saving final global model...")
    state_dict = result.arrays.to_torch_state_dict()
    save_path = "final_model.pt"
    torch.save(state_dict, save_path)
    print(f"✓ Saved: {save_path}")
    # models_dir = Path("models")
    # models_dir.mkdir(exist_ok=True)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # # Save final model
    # final_model_filename = f"sanet_fedavg_final_{partitioner_type}_r{num_rounds}_{timestamp}.pth"
    # final_model_path = models_dir / final_model_filename
    
    # final_state_dict = final_arrays.to_torch_state_dict()
    # torch.save({
    #     'model_state_dict': final_state_dict,
    #     'config': {
    #         'sa_channels': sa_channels,
    #         'num_rounds': num_rounds,
    #         'learning_rate': lr,
    #         'partitioner_type': partitioner_type,
    #         'timestamp': timestamp,
    #         'total_params': total_params,
    #     },
    #     'training_info': {
    #         'total_time': str(elapsed),
    #         'num_rounds': num_rounds,
    #         'final_mae': strategy.history['mae'][-1] if strategy.history['mae'] else None,
    #         'final_rmse': strategy.history['rmse'][-1] if strategy.history['rmse'] else None,
    #     }
    # }, final_model_path)
    
    # print(f"\n  ✓ Final model saved: {final_model_path}")
    # print(f"    File size: {final_model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # # Save best model (if different from final)
    # if strategy.best_mae < float('inf'):
    #     best_model_filename = f"sanet_fedavg_best_mae{strategy.best_mae:.2f}_r{strategy.best_round}_{timestamp}.pth"
    #     best_model_path = models_dir / best_model_filename
        
    #     # Note: We save final model as best since we don't track arrays per round
    #     # In production, you'd save arrays at each round
    #     torch.save({
    #         'model_state_dict': final_state_dict,
    #         'config': {
    #             'sa_channels': sa_channels,
    #             'num_rounds': num_rounds,
    #             'learning_rate': lr,
    #             'partitioner_type': partitioner_type,
    #             'timestamp': timestamp,
    #             'total_params': total_params,
    #         },
    #         'best_metrics': {
    #             'best_round': strategy.best_round,
    #             'best_mae': strategy.best_mae,
    #             'best_rmse': strategy.best_rmse,
    #         },
    #         'training_info': {
    #             'total_time': str(elapsed),
    #             'num_rounds': num_rounds,
    #         }
    #     }, best_model_path)
        
    #     print(f"\n  ✓ Best model saved: {best_model_path}")
    #     print(f"    Best MAE:  {strategy.best_mae:.2f} at round {strategy.best_round}")
    #     print(f"    Best RMSE: {strategy.best_rmse:.2f}")
    
    # # Save training history to JSON
    # history_filename = f"training_history_{partitioner_type}_r{num_rounds}_{timestamp}.json"
    # history_path = models_dir / history_filename
    
    # history_data = {
    #     'config': {
    #         'sa_channels': list(sa_channels),
    #         'num_rounds': num_rounds,
    #         'learning_rate': lr,
    #         'batch_size': batch_size,
    #         'partitioner_type': partitioner_type,
    #         'timestamp': timestamp,
    #     },
    #     'best_metrics': {
    #         'best_round': strategy.best_round,
    #         'best_mae': strategy.best_mae,
    #         'best_rmse': strategy.best_rmse,
    #     },
    #     'history': {
    #         'rounds': strategy.history['rounds'],
    #         'eval_loss': strategy.history['eval_loss'],
    #         'mae': strategy.history['mae'],
    #         'rmse': strategy.history['rmse'],
    #     },
    #     'training_info': {
    #         'total_time': str(elapsed),
    #         'total_params': total_params,
    #     }
    # }
    
    # with open(history_path, 'w') as f:
    #     json.dump(history_data, f, indent=2)
    
    # print(f"\n  ✓ Training history saved: {history_path}")
    
    # # Print final summary
    # print("\n" + "="*70)
    # print("📊 Training Summary")
    # print("="*70)
    # print(f"  Duration:           {elapsed}")
    # print(f"  Total rounds:       {num_rounds}")
    # print(f"  Total params:       {total_params:,}")
    # print(f"\n  Final Performance:")
    # if strategy.history['mae']:
    #     print(f"    Final MAE:        {strategy.history['mae'][-1]:.2f} persons")
    #     print(f"    Final RMSE:       {strategy.history['rmse'][-1]:.2f} persons")
    # print(f"\n  Best Performance:")
    # print(f"    Best MAE:         {strategy.best_mae:.2f} persons (Round {strategy.best_round})")
    # print(f"    Best RMSE:        {strategy.best_rmse:.2f} persons")
    
    # # Paper benchmark comparison
    # print(f"\n  Paper Benchmark (ShanghaiTech Part A):")
    # print(f"    SANet MAE:        ~67.0 persons")
    # print(f"    SANet RMSE:       ~104.5 persons")
    
    # # Performance comparison
    # if strategy.best_mae < float('inf'):
    #     mae_diff = strategy.best_mae - 67.0
    #     if mae_diff < 0:
    #         print(f"\n  🎉 Better than paper by {abs(mae_diff):.2f} MAE!")
    #     else:
    #         print(f"\n  📈 Gap from paper: {mae_diff:.2f} MAE")
    
    # print("="*70)
    
    print("="*70 + "\n")
    print("✅ Federated Learning completed successfully!")