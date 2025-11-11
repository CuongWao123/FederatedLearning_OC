"""fedavg-sanet: A Flower / PyTorch ServerApp."""

from typing import Any, Dict

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

# Dùng SANet làm model toàn cục
from fedavg_sanet.Task_SANET import SANet

# Default SANet channels theo paper
DEFAULT_SA_CHANNELS = (64, 128, 256, 512)

# Create ServerApp
app = ServerApp()


def _get_run_cfg(context: Context) -> Dict[str, Any]:
    """Lấy tham số từ run_config với giá trị mặc định an toàn."""
    rc = context.run_config

    # Các tham số cơ bản
    num_server_rounds = int(rc.get("num-server-rounds", 20))
    lr = float(rc.get("lr", 1e-4))
    local_epochs = int(rc.get("local-epochs", 1))

    # Cấu hình chiến lược
    fraction_train = float(rc.get("fraction-train", 1.0))      
    fraction_evaluate = float(rc.get("fraction-evaluate", 1.0))
    min_train_clients = int(rc.get("min-train-clients", 2))
    min_evaluate_clients = int(rc.get("min-evaluate-clients", 2))
    min_available_clients = int(rc.get("min-available-clients", 2))

    # Model config - parse dict to tuple
    sa_channels_cfg = rc.get("sa-channels", {"c1": 64, "c2": 128, "c3": 256, "c4": 512})
    if isinstance(sa_channels_cfg, dict):
        sa_channels = (sa_channels_cfg["c1"], sa_channels_cfg["c2"], 
                       sa_channels_cfg["c3"], sa_channels_cfg["c4"])
    else:
        sa_channels = tuple(sa_channels_cfg)

    return {
        "num_server_rounds": num_server_rounds,
        "lr": lr,
        "local_epochs": local_epochs,
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_clients": min_train_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "sa_channels": sa_channels,
    }


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    cfg = _get_run_cfg(context)

    print("\n" + "="*70)
    print("FedAvg-SANet Server Configuration")
    print("="*70)
    for k, v in cfg.items():
        print(f"  {k:25s}: {v}")
    print("="*70 + "\n")

    # 1) Khởi tạo global SANet model
    global_model = SANet(sa_channels=cfg["sa_channels"])
    
    # 2) Chuyển sang ArrayRecord (theo API mẫu của Flower)
    arrays = ArrayRecord(global_model.state_dict())

    # 3) Tạo FedAvg strategy
    strategy = FedAvg(fraction_train=cfg["fraction_train"])

    # 4) Tạo train config để gửi cho clients
    train_config = ConfigRecord({
        "lr": cfg["lr"],
        "local-epochs": cfg["local_epochs"],
    })

    # 5) Chạy federated learning (theo API mẫu)
    print(f"🚀 Starting FedAvg for {cfg['num_server_rounds']} rounds...\n")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=cfg["num_server_rounds"],
    )

    # 6) Lưu final model
    print("\n" + "="*70)
    print("💾 Saving final global model...")
    state_dict = result.arrays.to_torch_state_dict()
    save_path = "final_model.pt"
    torch.save(state_dict, save_path)
    print(f"✓ Saved: {save_path}")
    print("="*70 + "\n")
