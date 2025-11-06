"""ObjectCounting: A Flower / PyTorch app."""

from datetime import datetime
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from objectcounting.Task_2NN import Net2NN, count_parameters
from objectcounting.utils import (
    create_experiment_log_file,
    write_experiment_summary,
    LoggingFedAvg,
    save_model,
    print_config_summary
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config.get("local-epochs", 1)
    batch_size: int = context.run_config.get("batch-size", 10)
    
    # Read IID config with safe handling
    iid_value = context.run_config.get("iid", True)
    if isinstance(iid_value, str):
        iid = iid_value.lower() in ("true", "1", "yes")
    else:
        iid = bool(iid_value)

    # Build config dict
    config = {
        "fraction_train": fraction_train,
        "num_rounds": num_rounds,
        "lr": lr,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "iid": iid
    }

    # Load and verify model
    global_model = Net2NN()
    total_params = count_parameters(global_model)
    assert total_params == 199_210, f"Expected 199,210 params, got {total_params:,}"
    
    # Print config summary to console
    print_config_summary(config, total_params)
    
    # Create experiment log file
    log_file = create_experiment_log_file(config)
    print(f"\n📝 Logging to: {log_file}\n")
    
    # Create model arrays
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with logging
    strategy = LoggingFedAvg(
        fraction_train=fraction_train,
        log_file_path=log_file
    )
    
    # Log strategy configuration to file
    strategy.summary()

    # Start federated learning
    print("🚀 Starting Federated Learning...\n")
    start_time = datetime.now()
    
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "lr": lr,
            "local-epochs": local_epochs,
            "iid": iid,
            "batch-size": batch_size
        }),
        num_rounds=num_rounds,
    )
    
    end_time = datetime.now()

    # Write experiment summary to log file
    write_experiment_summary(
        log_file=log_file,
        train_metrics=strategy.train_metrics,
        eval_metrics=strategy.eval_metrics,
        start_time=start_time,
        end_time=end_time,
        target_accuracy=0.97
    )

    # Save final model
    print("\n" + "=" * 60)
    print("💾 Saving final model...")
    state_dict = result.arrays.to_torch_state_dict()
    model_path = save_model(state_dict, config)
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Log saved: {log_file}")
    print("=" * 60)
    
    # Print final summary
    elapsed = end_time - start_time
    print(f"\n📊 Experiment Summary:")
    print(f"  Duration: {elapsed}")
    print(f"  Total Rounds: {len(strategy.train_metrics)}")
    
    # Check target accuracy
    rounds_to_97 = None
    for round_num in sorted(strategy.eval_metrics.keys()):
        if strategy.eval_metrics[round_num]["eval_acc"] >= 0.97:
            rounds_to_97 = round_num
            break
    
    if rounds_to_97:
        final_acc = strategy.eval_metrics[rounds_to_97]["eval_acc"]
        print(f"  ✓ Reached 97% accuracy at round: {rounds_to_97}")
        print(f"    Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    else:
        if strategy.eval_metrics:
            final_round = max(strategy.eval_metrics.keys())
            final_acc = strategy.eval_metrics[final_round]["eval_acc"]
            print(f"  ✗ Did not reach 97% accuracy")
            print(f"    Best: {final_acc:.4f} ({final_acc*100:.2f}%) at round {final_round}")