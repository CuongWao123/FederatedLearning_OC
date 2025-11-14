"""SANet Crowd Counting: Flower Client App for Federated Learning."""

import os
# Force GPU 1 usage
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import numpy as np
# Import từ SANET.py ở thư mục gốc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SANET import SANet, load_data, train, test_fn, get_device
import time
# Flower ClientApp
app = ClientApp()


# @app.train()
# def train_client(msg: Message, context: Context):
#     """
#     Train SANet model on local crowd counting data.
    
#     Paper: Scale Aggregation Network (ECCV 2018)
#     - Architecture: SANet with 4 SA modules
#     - Loss: MSE + α*SSIM + β*Count (α=1e-3, β=1e-3)
#     - Optimizer: Adam with lr=1e-5
#     """
    
#     # Get SA channels config from server (list) and convert to tuple
#     sa_channels_list = context.run_config.get("sa-channels", [64, 128, 256, 512])
#     sa_channels = tuple(sa_channels_list)  # Convert list back to tuple
    
#     # Initialize SANet model
#     model = SANet(sa_channels=sa_channels)
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

#     # Get node configuration (partition ID)
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
    
#     # Get training configuration from server
#     csv_path = context.run_config["csv-path"]
#     batch_size = context.run_config.get("batch-size", 1)
#     sigma = context.run_config.get("sigma", 4)
#     partitioner_type = context.run_config.get("partitioner-type", "iid")
#     dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
#     num_workers = context.run_config.get("num-workers", 0)
#     pin_memory = context.run_config.get("pin-memory", False)

#     # GPU device (cuda:0 = physical GPU 1 due to CUDA_VISIBLE_DEVICES='1')
#     device = get_device(gpu_id=0)

#     # Load local partition data
#     trainloader, _ = load_data(
#         partition_id=partition_id,
#         num_partitions=num_partitions,
#         csv_path=csv_path,
#         batch_size=batch_size,
#         sigma=sigma,
#         partitioner_type=partitioner_type,
#         dirichlet_alpha=dirichlet_alpha,
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )

#     # Training hyperparameters
#     local_epochs = context.run_config["local-epochs"]
#     lr = context.run_config["lr"]
#     amp = context.run_config.get("amp", True)
#     grad_clip = context.run_config.get("grad-clip", 0.0)
#     alpha = context.run_config.get("alpha", 1e-3)  # SSIM loss weight
#     beta = context.run_config.get("beta", 1e-3)    # Count loss weight

#     # Train the model (cuda:0 = physical GPU 1)
#     train_loss = train(
#         net=model,
#         trainloader=trainloader,
#         epochs=local_epochs,
#         lr=lr,
#         device=device,
#         amp=amp,
#         grad_clip=grad_clip,
#         alpha=alpha,
#         beta=beta
#     )

#     # Return updated model weights and metrics
#     model_record = ArrayRecord(model.state_dict())
#     metrics = {
#         "train_loss": train_loss,
#         "num-examples": len(trainloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
#     return Message(content=content, reply_to=msg)
@app.train()
def train_client(msg: Message, context: Context):
    # ...existing code...
       
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
    start_time = time.time()  # Bắt đầu thời gian
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
    inference_time = time.time() - start_time  # Tính thời gian inference

    # Return updated model weights and metrics
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "inference_time": inference_time,  # Thêm inference time
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)
# ...existing code...
import numpy as np
# ...existing code...

def _sanitize_metrics(metric_dict: dict) -> dict:
    """Convert None / numpy types to plain int/float or lists. Replace invalid with NaN."""
    out = {}
    for k, v in metric_dict.items():
        # None -> NaN
        if v is None:
            out[k] = float("nan")
            continue

        # numpy scalar
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if np.issubdtype(type(v), np.floating) else int(v)
            continue

        # numpy array / list / tuple -> list of floats/ints
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v)
            if arr.dtype.kind in ("f", "c"):
                out[k] = arr.astype(float).tolist()
            else:
                out[k] = arr.astype(int).tolist()
            continue

        # plain int/float
        if isinstance(v, (int, float)):
            out[k] = v
            continue

        # fallback: try cast to float, else NaN
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out



@app.evaluate()
def evaluate_client(msg: Message, context: Context):
    """
    Evaluate SANet model on local test data.
    
    Metrics:
    - eval_loss: Combined loss (MSE + SSIM + Count)
    - mae: Mean Absolute Error (paper metric)
    - rmse: Root Mean Square Error (paper metric)
    """
    sa_channels_list = context.run_config.get("sa-channels", [64, 128, 256, 512])
    sa_channels = tuple(sa_channels_list)
    model = SANet(sa_channels=sa_channels)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = get_device(gpu_id=0)
    model.to(device)
    model.eval()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    csv_path = context.run_config["csv-path"]
    batch_size = context.run_config.get("batch-size", 1)
    sigma = context.run_config.get("sigma", 4)
    partitioner_type = context.run_config.get("partitioner-type", "iid")
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
    num_workers = context.run_config.get("num-workers", 0)
    pin_memory = context.run_config.get("pin-memory", False)

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

    alpha = context.run_config.get("alpha", 1e-3)
    beta = context.run_config.get("beta", 1e-3)

    # Use provided test_fn for eval_loss, mae, rmse (keeps existing behavior)
    eval_loss, mae, rmse = test_fn(
        net=model,
        valloader=testloader,
        device=device,
        alpha=alpha,
        beta=beta
    )

    # Compute R2 and detailed inference time + memory
    y_true_counts = []
    y_pred_counts = []
    total_infer_start = time.time()
    with torch.no_grad():
        for imgs, dms in testloader:
            imgs = imgs.to(device)
            # Forward
            preds = model(imgs)
            # Sum per-sample counts: flatten then sum
            preds_counts = preds.view(preds.size(0), -1).sum(dim=1).cpu().numpy()
            gt_counts = dms.view(dms.size(0), -1).sum(dim=1).cpu().numpy()
            y_pred_counts.extend(preds_counts.tolist())
            y_true_counts.extend(gt_counts.tolist())
    total_infer_time = time.time() - total_infer_start
    total_samples = len(y_true_counts) if y_true_counts else len(testloader.dataset)

    # R2 (safe)
    y_true_arr = np.array(y_true_counts, dtype=np.float64)
    y_pred_arr = np.array(y_pred_counts, dtype=np.float64)
    denom = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    if denom <= 0:
        r2_score = float("nan")
    else:
        r2_score = 1.0 - (np.sum((y_true_arr - y_pred_arr) ** 2) / denom)

    # Memory (GPU preferred, fallback to psutil for CPU)
    memory_mb = None
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            memory_bytes = torch.cuda.max_memory_allocated(device) or torch.cuda.memory_allocated(device)
            memory_mb = float(memory_bytes) / (1024 ** 2)
        else:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        memory_mb = None

    # Per-sample inference time
    infer_time_per_sample = total_infer_time / total_samples if total_samples > 0 else None

    metrics = {
        "eval_loss": eval_loss,
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score) if not np.isnan(r2_score) else None,
        "inference_time_total": float(total_infer_time),
        "inference_time_per_sample": float(infer_time_per_sample) if infer_time_per_sample is not None else None,
        "memory_mb": float(memory_mb) if memory_mb is not None else None,
        "num-examples": int(total_samples),
    }

    # Sanitize before creating MetricRecord
    metrics = _sanitize_metrics(metrics)
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
    
    # Evaluate the model
    # eval_loss, mae, rmse = test_fn(
    #     net=model,
    #     valloader=testloader,
    #     device=device,
    #     alpha=alpha,
    #     beta=beta
    # )

    # # Return evaluation metrics
    # metrics = {
    #     "test": 0,
    #     "eval_loss": eval_loss,
    #     "mae": mae,
    #     "rmse": rmse,
    #     "num-examples": len(testloader.dataset),
    # }
    # metric_record = MetricRecord(metrics)
    # content = RecordDict({"metrics": metric_record})
    
    # return Message(content=content, reply_to=msg)
