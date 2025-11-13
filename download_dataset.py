import kagglehub
import os

save_dir = "datasets" 
os.makedirs(save_dir, exist_ok=True)
path = kagglehub.dataset_download("tthien/shanghaitech", path=save_dir)

print("Dataset saved to:", path)

