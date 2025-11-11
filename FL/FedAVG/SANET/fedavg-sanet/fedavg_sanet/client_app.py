"""fedavg-sanet: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedavg_sanet.Task_SANET import SANet
from fedavg_sanet.Task_SANET import test_fn
from fedavg_sanet.Task_SANET import train as train_fn
from fedavg_sanet.DatasetLoader import load_data

# Flower ClientApp
app = ClientApp()


def get_device():
    """Detect and configure GPU device for Ray simulation."""
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available - using CPU")
        return torch.device("cpu")
    
    try:
        import ray
        if ray.is_initialized():
            # In Ray simulation, get assigned GPU
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                device = torch.device(f"cuda:{gpu_ids[0]}")
                torch.cuda.set_device(device)
                print(f"🔥 Ray assigned GPU: {device}")
            else:
                # Ray initialized but no GPU assigned, use cuda:0
                device = torch.device("cuda:0")
                print(f"🔥 Using default GPU: {device}")
        else:
            # Not in Ray, use cuda:0
            device = torch.device("cuda:0")
            print(f"🔥 Using GPU: {device}")
        
        print(f"   GPU: {torch.cuda.get_device_name(device)}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        return device
    except Exception as e:
        print(f"⚠️ Error detecting GPU: {e}")
        print("   Falling back to CPU")
        return torch.device("cpu")


@app.train()
def train(msg: Message, context: Context):
    """Train model (crowd counting) trên local partition."""
    
    print(f"\n{'='*60}")
    # 1) Detect and configure device
    device = get_device()
    print(f"{'='*60}\n")
    
    # 2) Model & weights
    model = SANet(sa_channels=(64,128,256,512)) 
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    print(f"✅ Model loaded on: {next(model.parameters()).device}\n")

    # 2) Đọc tham số train từ run_config
    rc = context.run_config
    lr            = float(rc.get("lr", 1e-4))
    local_epochs  = int(rc.get("local-epochs", 1))
    batch_size    = int(rc.get("batch_size", 8))
    sigma         = int(rc.get("sigma", 4))
    target_size   = tuple(rc.get("target_size", (384, 512)))
    amp           = bool(rc.get("amp", True)) and device.type == "cuda"  # Only AMP if GPU
    grad_clip     = float(rc.get("grad_clip", 0.0))
    
    # Loss parameters
    alpha         = float(rc.get("alpha", 1e-3))
    beta          = float(rc.get("beta", 1e-3))
    ssim_window   = int(rc.get("ssim_window", 11))
    ssim_sigma    = float(rc.get("ssim_sigma", 1.5))

    # 3) Load data (theo partition)
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Nếu load_data cần csv_path/… thì lấy từ run_config
    csv_path     = rc.get("csv_path", "shanghaitech_train.csv")
    part_type    = rc.get("partitioner_type", "iid")
    alpha        = float(rc.get("dirichlet_alpha", 0.5))
    num_workers  = int(rc.get("num_workers", 0))
    pin_memory   = bool(rc.get("pin_memory", True)) and device.type == "cuda"  # Only pin_memory if GPU

    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        csv_path=csv_path,
        batch_size=batch_size,
        sigma=sigma,
        partitioner_type=part_type,
        dirichlet_alpha=alpha,
        num_workers=num_workers,
        pin_memory=pin_memory,
        target_size=target_size,
    )
    
    print(f"📊 Training data loaded: {len(trainloader.dataset)} samples")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {local_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   AMP: {amp}")
    print(f"   Pin memory: {pin_memory}\n")

    # 4) Train
    train_loss = train_fn(
        model,
        trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        amp=amp,
        grad_clip=grad_clip,
        alpha=alpha,
        beta=beta,
        ssim_window=ssim_window,
        ssim_sigma=ssim_sigma,
    )
    
    print(f"\n✅ Training completed. Loss: {train_loss:.6f}")
    if device.type == "cuda":
        print(f"   GPU Memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")

    # 5) Trả model & metrics
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate model (crowd counting) trên local partition."""
    
    print(f"\n{'='*60}")
    # 1) Detect and configure device
    device = get_device()
    print(f"{'='*60}\n")
    
    # 2) Model & weights
    model = SANet(sa_channels=(64,128,256,512))  
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    print(f"✅ Model loaded on: {next(model.parameters()).device}\n")

    # 2) Tham số eval
    rc = context.run_config
    batch_size  = int(rc.get("batch_size", 8))
    sigma       = int(rc.get("sigma", 4))
    target_size = tuple(rc.get("target_size", (384, 512)))
    
    # Loss parameters
    alpha       = float(rc.get("alpha", 1e-3))
    beta        = float(rc.get("beta", 1e-3))
    ssim_window = int(rc.get("ssim_window", 11))
    ssim_sigma  = float(rc.get("ssim_sigma", 1.5))

    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    csv_path     = rc.get("csv_path", "shanghaitech_train.csv")
    part_type    = rc.get("partitioner_type", "iid")
    alpha        = float(rc.get("dirichlet_alpha", 0.5))
    num_workers  = int(rc.get("num_workers", 0))
    pin_memory   = bool(rc.get("pin_memory", True)) and device.type == "cuda"  # Only pin_memory if GPU

    # 3) Data (dùng split 80/20 trong partition -> "testloader" là 20%)
    _, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        csv_path=csv_path,
        batch_size=batch_size,
        sigma=sigma,
        partitioner_type=part_type,
        dirichlet_alpha=alpha,
        num_workers=num_workers,
        pin_memory=pin_memory,
        target_size=target_size,
    )
    
    print(f"📊 Validation data loaded: {len(valloader.dataset)} samples\n")

    # 4) Evaluate
    eval_loss, eval_mae, eval_rmse = test_fn(
        model, 
        valloader, 
        device,
        alpha=alpha,
        beta=beta,
        ssim_window=ssim_window,
        ssim_sigma=ssim_sigma,
    )
    
    print(f"\n✅ Evaluation completed:")
    print(f"   Loss: {eval_loss:.6f}")
    print(f"   MAE: {eval_mae:.2f}")
    print(f"   RMSE: {eval_rmse:.2f}\n")

    # 5) Trả metrics (đổi tên khoá cho rõ nghĩa)
    metrics = {
        "eval_loss": eval_loss,
        "eval_mae": eval_mae,
        "eval_rmse": eval_rmse,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
