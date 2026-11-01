# ============================================================================
# APPROACH 2: training a separate model per task
#
# Second version of the training script: instead of one model for everything,
# train one RobotGPT for "pick" and one for "reach". Each model only ever sees
# its own task, so the header and start token lines are skipped. Servo values
# (-1800..1800, tenths of a degree) are scaled to 0..1 before training.
# Saves model_<task>.pt and logs the loss curve to Weights & Biases.
# ============================================================================

import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from gpt import RobotGPT

# LSS Servo Raw Unit Constants (-1800.0 to +1800.0)
SERVO_MIN = -1800.0
SERVO_RANGE = 3600.0  # 1800.0 - (-1800.0)

# Loads every demonstration for one task and normalizes it to 0..1.
class SingleTaskDataset(Dataset):
    def __init__(self, folder_path, max_seq_len=200):
        self.samples = []
        self.max_seq_len = max_seq_len

        file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
        print(f"Loading {len(file_paths)} trajectory files from '{folder_path}'...")

        for path in file_paths:
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            # Skip invalid or empty files
            if len(lines) <= 2:
                continue

            # SKIP LINES 1 & 2 (Header & Start Token)
            # Only process continuous raw servo trajectory positions (Lines 3+)
            raw_trajectory = lines[2:]
            
            # Normalize raw LSS positions (e.g., -1003) into [0.0, 1.0] range
            trajectory = [
                [(float(x) - SERVO_MIN) / SERVO_RANGE for x in line.split()] 
                for line in raw_trajectory
            ]

            if len(trajectory) > self.max_seq_len:
                trajectory = trajectory[:self.max_seq_len]

            tensor_seq = torch.tensor(trajectory, dtype=torch.float32)
            self.samples.append(tensor_seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        # Input x: step 0 to T-1
        # Target y: step 1 to T
        return seq[:-1], seq[1:]

def pad_collate(batch):
    x_list, y_list = zip(*batch)
    lengths = [len(x) for x in x_list]
    max_len = max(lengths)

    padded_x = torch.zeros(len(x_list), max_len, 5)
    padded_y = torch.zeros(len(y_list), max_len, 5)

    for i in range(len(x_list)):
        padded_x[i, :lengths[i]] = x_list[i]
        padded_y[i, :lengths[i]] = y_list[i]

    return padded_x, padded_y

# Train and save one model for one task.
def train_model_for_task(task_name):
    config = {
        "epochs": 200,
        "batch_size": 4,
        "learning_rate": 5e-4,
        "input_dim": 5,
        "d_model": 128,
        "n_head": 4,
        "n_layer": 4,
        "max_seq_len": 200,
        "task_name": task_name
    }

    folder_path = os.path.join("dataset", task_name)
    if not os.path.exists(folder_path):
        print(f"[ERROR] Directory '{folder_path}' does not exist.")
        return

    wandb.init(project="robot-gpt-separate", config=config, name=f"run_{task_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n==========================================")
    print(f"       TRAINING MODEL FOR: {task_name.upper()}")
    print(f"==========================================")

    dataset = SingleTaskDataset(folder_path=folder_path, max_seq_len=config["max_seq_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=pad_collate)

    model = RobotGPT(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        max_seq_len=config["max_seq_len"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    model.train()
    for epoch in range(1, config["epochs"] + 1):
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

            loss = F.mse_loss(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})

        if epoch % 20 == 0 or epoch == 1:
            print(f"[{task_name.upper()}] Epoch {epoch}/{config['epochs']} | Normalized MSE Loss: {avg_loss:.6f}")

    save_path = f"model_{task_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Saved model weights to '{save_path}'")
    
    artifact = wandb.Artifact(f'robot-gpt-{task_name}', type='model')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    # Train both models sequentially
    train_model_for_task("pick")
    train_model_for_task("reach")


