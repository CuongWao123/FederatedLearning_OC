"""Run MNIST 2NN experiments matching Table 1 from FedAvg paper."""

import subprocess
import time
from datetime import datetime
from pathlib import Path
import re

# Experiments from Table 1: 2NN, E=1
# Target accuracy: 97%
TARGET_ACCURACY = 0.97

experiments_2nn = [
    # IID with B=10
    {
        "name": "2NN_IID_C0.0_B10",
        "config": "fraction-train=0.01 num-server-rounds=500 local-epochs=1 lr=0.01 iid=true",
        "expected_rounds": 316,  # From table
        "description": "IID, C=0.0 (1 client), B=10"
    },
    {
        "name": "2NN_IID_C0.1_B10",
        "config": "fraction-train=0.1 num-server-rounds=150 local-epochs=1 lr=0.01 iid=true",
        "expected_rounds": 87,
        "description": "IID, C=0.1 (10 clients), B=10"
    },
    {
        "name": "2NN_IID_C0.2_B10",
        "config": "fraction-train=0.2 num-server-rounds=120 local-epochs=1 lr=0.01 iid=true",
        "expected_rounds": 77,
        "description": "IID, C=0.2 (20 clients), B=10"
    },
    {
        "name": "2NN_IID_C0.5_B10",
        "config": "fraction-train=0.5 num-server-rounds=120 local-epochs=1 lr=0.01 iid=true",
        "expected_rounds": 75,
        "description": "IID, C=0.5 (50 clients), B=10"
    },
    {
        "name": "2NN_IID_C1.0_B10",
        "config": "fraction-train=1.0 num-server-rounds=120 local-epochs=1 lr=0.01 iid=true",
        "expected_rounds": 70,
        "description": "IID, C=1.0 (100 clients), B=10"
    },
    
    # Non-IID with B=10
    {
        "name": "2NN_NonIID_C0.0_B10",
        "config": "fraction-train=0.01 num-server-rounds=3500 local-epochs=1 lr=0.01 iid=false",
        "expected_rounds": 3275,
        "description": "Non-IID, C=0.0 (1 client), B=10"
    },
    {
        "name": "2NN_NonIID_C0.1_B10",
        "config": "fraction-train=0.1 num-server-rounds=900 local-epochs=1 lr=0.01 iid=false",
        "expected_rounds": 664,
        "description": "Non-IID, C=0.1 (10 clients), B=10"
    },
    {
        "name": "2NN_NonIID_C0.2_B10",
        "config": "fraction-train=0.2 num-server-rounds=800 local-epochs=1 lr=0.01 iid=false",
        "expected_rounds": 619,
        "description": "Non-IID, C=0.2 (20 clients), B=10"
    },
    {
        "name": "2NN_NonIID_C0.5_B10",
        "config": "fraction-train=0.5 num-server-rounds=600 local-epochs=1 lr=0.01 iid=false",
        "expected_rounds": 443,
        "description": "Non-IID, C=0.5 (50 clients), B=10"
    },
    {
        "name": "2NN_NonIID_C1.0_B10",
        "config": "fraction-train=1.0 num-server-rounds=500 local-epochs=1 lr=0.01 iid=false",
        "expected_rounds": 380,
        "description": "Non-IID, C=1.0 (100 clients), B=10"
    },
]


def extract_metrics_from_output(output):
    """Extract accuracy and rounds from command output."""
    # Look for accuracy patterns in output
    accuracy_pattern = r"accuracy[:\s]+([0-9.]+)"
    round_pattern = r"Round (\d+)"
    
    accuracies = re.findall(accuracy_pattern, output, re.IGNORECASE)
    rounds = re.findall(round_pattern, output)
    
    if accuracies and rounds:
        # Convert to float and find when target accuracy is reached
        acc_values = [float(a) for a in accuracies]
        round_values = [int(r) for r in rounds]
        
        for i, acc in enumerate(acc_values):
            if acc >= TARGET_ACCURACY:
                return round_values[i] if i < len(round_values) else None, acc
    
    return None, max([float(a) for a in accuracies]) if accuracies else 0.0


def run_experiment(exp, log_dir):
    """Run a single experiment and log results."""
    print(f"\n{'='*70}")
    print(f"Experiment: {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Expected rounds to 97% accuracy: {exp['expected_rounds']}")
    print(f"Config: {exp['config']}")
    print(f"{'='*70}\n")
    
    cmd = f"flwr run . --run-config \"{exp['config']}\""
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    # Extract metrics
    rounds_to_target, max_accuracy = extract_metrics_from_output(result.stdout)
    
    # Determine status
    if result.returncode == 0:
        if rounds_to_target:
            status = f"SUCCESS (reached 97% at round {rounds_to_target})"
            achieved_target = True
        else:
            status = f"COMPLETED (max accuracy: {max_accuracy:.2%}, target not reached)"
            achieved_target = False
    else:
        status = "FAILED"
        achieved_target = False
    
    # Save detailed logs
    log_file = log_dir / f"{exp['name']}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp['name']}\n")
        f.write(f"Description: {exp['description']}\n")
        f.write(f"Expected rounds: {exp['expected_rounds']}\n")
        f.write(f"Actual rounds to 97%: {rounds_to_target if rounds_to_target else 'N/A'}\n")
        f.write(f"Max accuracy achieved: {max_accuracy:.2%}\n")
        f.write(f"Config: {exp['config']}\n")
        f.write(f"Start time: {datetime.now()}\n")
        f.write(f"Duration: {elapsed/60:.2f} minutes\n")
        f.write(f"Status: {status}\n")
        f.write(f"\n{'='*70}\n")
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write(f"\n{'='*70}\n")
        f.write("STDERR:\n")
        f.write(result.stderr)
    
    print(f"\n✓ {exp['name']} completed in {elapsed/60:.2f} minutes")
    if rounds_to_target:
        speedup = exp['expected_rounds'] / rounds_to_target if rounds_to_target > 0 else 0
        print(f"  Reached 97% accuracy at round {rounds_to_target} (expected: {exp['expected_rounds']})")
        print(f"  Speedup: {speedup:.1f}x")
    else:
        print(f"  Max accuracy: {max_accuracy:.2%} (target 97% not reached)")
    print(f"  Log saved: {log_file}")
    
    return {
        "success": result.returncode == 0,
        "achieved_target": achieved_target,
        "rounds": rounds_to_target,
        "max_accuracy": max_accuracy,
        "elapsed": elapsed
    }


def main():
    """Run all 2NN experiments."""
    print("="*70)
    print("MNIST 2NN Federated Learning Experiments")
    print("Replicating Table 1 from FedAvg paper (McMahan et al., 2017)")
    print("="*70)
    print(f"\nTotal experiments: {len(experiments_2nn)}")
    print(f"Target: {TARGET_ACCURACY:.0%} test accuracy")
    print("Model: 2NN (199,210 parameters)")
    print("Local epochs: E=1")
    print("Batch size: B=10")
    print("="*70)
    
    # Create log directory
    log_dir = Path("experiment_logs_2nn")
    log_dir.mkdir(exist_ok=True)
    
    # Run experiments
    results = {}
    total_start = time.time()
    
    for i, exp in enumerate(experiments_2nn, 1):
        print(f"\n[{i}/{len(experiments_2nn)}] Starting: {exp['name']}")
        
        try:
            result = run_experiment(exp, log_dir)
            results[exp['name']] = {
                "status": "✓" if result["achieved_target"] else "✗",
                "time": result["elapsed"],
                "expected_rounds": exp['expected_rounds'],
                "actual_rounds": result["rounds"],
                "max_accuracy": result["max_accuracy"]
            }
        except Exception as e:
            print(f"❌ Error in {exp['name']}: {e}")
            results[exp['name']] = {
                "status": "ERROR",
                "time": 0,
                "expected_rounds": exp['expected_rounds'],
                "actual_rounds": None,
                "max_accuracy": 0.0
            }
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY - Comparison with Table 1")
    print("="*70)
    print(f"{'Experiment':<25} {'Status':<8} {'Expected':<10} {'Actual':<10} {'Accuracy':<10} {'Time(min)'}")
    print("-"*70)
    
    for name, result in results.items():
        time_str = f"{result['time']/60:.1f}" if result['time'] > 0 else "N/A"
        actual_str = str(result['actual_rounds']) if result['actual_rounds'] else "N/A"
        acc_str = f"{result['max_accuracy']:.2%}" if result['max_accuracy'] > 0 else "N/A"
        
        print(f"{name:<25} {result['status']:<8} {result['expected_rounds']:<10} {actual_str:<10} {acc_str:<10} {time_str}")
    
    print("="*70)
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"Logs saved in: {log_dir.absolute()}")
    
    # Calculate statistics
    successful = sum(1 for r in results.values() if r['status'] == '✓')
    print(f"\nSuccessfully reached target: {successful}/{len(experiments_2nn)}")
    print("="*70)
    
    # Save summary
    summary_file = log_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("MNIST 2NN Experiments Summary - Table 1 Replication\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Target Accuracy: {TARGET_ACCURACY:.0%}\n")
        f.write(f"Total time: {total_elapsed/3600:.2f} hours\n\n")
        f.write(f"{'Experiment':<25} {'Status':<8} {'Expected':<10} {'Actual':<10} {'Accuracy':<10} {'Time(min)'}\n")
        f.write("-"*100 + "\n")
        for name, result in results.items():
            time_str = f"{result['time']/60:.1f}" if result['time'] > 0 else "N/A"
            actual_str = str(result['actual_rounds']) if result['actual_rounds'] else "N/A"
            acc_str = f"{result['max_accuracy']:.2%}" if result['max_accuracy'] > 0 else "N/A"
            f.write(f"{name:<25} {result['status']:<8} {result['expected_rounds']:<10} {actual_str:<10} {acc_str:<10} {time_str}\n")
        
        f.write(f"\nSuccessfully reached target: {successful}/{len(experiments_2nn)}")
    
    print(f"\nSummary saved: {summary_file}")


if __name__ == "__main__":
    main()