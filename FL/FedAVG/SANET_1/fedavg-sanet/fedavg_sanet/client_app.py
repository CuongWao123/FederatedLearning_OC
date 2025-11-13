"""SANet Crowd Counting: Flower Client App for Federated Learning."""

import os
# Force GPU 1 usage
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Import từ SANET.py ở thư mục gốc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SANET import SANet, load_data, train, test_fn, get_device

# Flower ClientApp
app = ClientApp()


@app.train()
def train_client(msg: Message, context: Context):
    """
    Train SANet model on local crowd counting data.
    
    Paper: Scale Aggregation Network (ECCV 2018)
    - Architecture: SANet with 4 SA modules
    - Loss: MSE + α*SSIM + β*Count (α=1e-3, β=1e-3)
    - Optimizer: Adam with lr=1e-5
    """
    
    # Get SA channels config from server (list) and convert to tuple
    sa_channels_list = context.run_config.get("sa-channels", [64, 128, 256, 512])
    sa_channels = tuple(sa_channels_list)  # Convert list back to tuple
    
    # Initialize SANet model
    model = SANet(sa_channels=sa_channels)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Get node configuration (partition ID)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Get training configuration from server
    csv_path = context.run_config["csv-path"]
    batch_size = context.run_config.get("batch-size", 1)
    sigma = context.run_config.get("sigma", 4)
    partitioner_type = context.run_config.get("partitioner-type", "iid")
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
    num_workers = context.run_config.get("num-workers", 0)
    pin_memory = context.run_config.get("pin-memory", False)

    # GPU device (cuda:0 = physical GPU 1 due to CUDA_VISIBLE_DEVICES='1')
    device = get_device(gpu_id=0)

    # Load local partition data
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        csv_path=csv_path,
        batch_size=batch_size,
        sigma=sigma,
        partitioner_type=partitioner_type,
        dirichlet_alpha=dirichlet_alpha,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Training hyperparameters
    local_epochs = context.run_config["local-epochs"]
    lr = context.run_config["lr"]
    amp = context.run_config.get("amp", True)
    grad_clip = context.run_config.get("grad-clip", 0.0)
    alpha = context.run_config.get("alpha", 1e-3)  # SSIM loss weight
    beta = context.run_config.get("beta", 1e-3)    # Count loss weight

    # Train the model (cuda:0 = physical GPU 1)
    train_loss = train(
        net=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        amp=amp,
        grad_clip=grad_clip,
        alpha=alpha,
        beta=beta
    )

    # Return updated model weights and metrics
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context):
    """
    Evaluate SANet model on local test data.
    
    Metrics:
    - eval_loss: Combined loss (MSE + SSIM + Count)
    - mae: Mean Absolute Error (paper metric)
    - rmse: Root Mean Square Error (paper metric)
    """
    
    # Get SA channels config from server (list) and convert to tuple
    sa_channels_list = context.run_config.get("sa-channels", [64, 128, 256, 512])
    sa_channels = tuple(sa_channels_list)  # Convert list back to tuple
    
    # Initialize SANet model
    model = SANet(sa_channels=sa_channels)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # GPU device (cuda:0 = physical GPU 1 due to CUDA_VISIBLE_DEVICES='1')
    device = get_device(gpu_id=0)
    model.to(device)

    # Get node configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Get evaluation configuration
    csv_path = context.run_config["csv-path"]
    batch_size = context.run_config.get("batch-size", 1)
    sigma = context.run_config.get("sigma", 4)
    partitioner_type = context.run_config.get("partitioner-type", "iid")
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
    num_workers = context.run_config.get("num-workers", 0)
    pin_memory = context.run_config.get("pin-memory", False)

    # Load local test data
    _, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        csv_path=csv_path,
        batch_size=batch_size,
        sigma=sigma,
        partitioner_type=partitioner_type,
        dirichlet_alpha=dirichlet_alpha,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Loss hyperparameters
    alpha = context.run_config.get("alpha", 1e-3)
    beta = context.run_config.get("beta", 1e-3)

    # Evaluate the model
    eval_loss, mae, rmse = test_fn(
        net=model,
        valloader=testloader,
        device=device,
        alpha=alpha,
        beta=beta
    )

    # Return evaluation metrics
    metrics = {
        "eval_loss": eval_loss,
        "mae": mae,
        "rmse": rmse,
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)
