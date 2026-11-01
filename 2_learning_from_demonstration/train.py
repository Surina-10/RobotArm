# ============================================================================
# APPROACH 2: training one model on every task
#
# Loads all recorded demonstrations from dataset/*/, normalizes the servo
# values, and trains RobotGPT to predict each next time step from the steps
# before it (next-step prediction, like a language model). The task header
# and start token are the first two inputs, so one model can learn several
# tasks. Loss curves are logged to Weights & Biases (wandb).
# ============================================================================

import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from gpt import RobotGPT

# Normalization constants (assumes max servo angle is 180 degrees)
SERVO_MAX = 180.0
START_TOKEN_VAL = 999.0  # Kept unnormalized as explicit marker

# Reads every trajectory file into a sequence: [header, start token, step1, step2, ...].
class RobotTrajectoryDataset(Dataset):
    def __init__(self, dataset_dir="dataset", max_seq_len=200):
        self.samples = []
        self.max_seq_len = max_seq_len

        file_paths = glob.glob(os.path.join(dataset_dir, "*", "*.txt"))
        print(f"Loading {len(file_paths)} trajectory files...")

        for path in file_paths:
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) < 3:
                continue

            # Line 1: Header (Unscaled binary indicator)
            header = [float(x) for x in lines[0].split()]
            
            # Line 2: Start Token
            start_token = [float(x) for x in lines[1].split()]
            
            # Lines 3+: Servo Positions (Normalized to [0.0, 1.0])
            trajectory = [[float(x) / SERVO_MAX for x in line.split()] for line in lines[2:]]

            full_seq = [header, start_token] + trajectory

            if len(full_seq) > self.max_seq_len:
                full_seq = full_seq[:self.max_seq_len]
            
            tensor_seq = torch.tensor(full_seq, dtype=torch.float32)
            self.samples.append(tensor_seq)

    def __len__(self):
        return len(self.samples)

    # Input = steps 0..T-1, target = steps 1..T, so every step learns to predict the next.
    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y

# Demonstrations have different lengths; pad them with zeros so they fit in one batch.
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

def train():
    # Training settings.
    config = {
        "epochs": 150,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "input_dim": 5,
        "d_model": 128,
        "n_head": 4,
        "n_layer": 4,
        "max_seq_len": 200,
        "dataset_dir": "dataset"
    }

    wandb.init(
        project="robot-gpt-training",
        config=config,
        name="masked_normalized_run"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    dataset = RobotTrajectoryDataset(dataset_dir=config["dataset_dir"], max_seq_len=config["max_seq_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=pad_collate)

    model = RobotGPT(
        input_dim=config["input_dim"], 
        d_model=config["d_model"], 
        n_head=config["n_head"], 
        n_layer=config["n_layer"], 
        max_seq_len=config["max_seq_len"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    wandb.watch(model, log="all", log_freq=10)

    global_step = 0
    model.train()

    for epoch in range(1, config["epochs"] + 1):
        total_epoch_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            # MASKING: Calculate loss starting ONLY from timestep index 2 onward.
            # Index 0 predicts the Start Token from the Header.
            # Index 1 predicts the first servo step from the Start Token.
            # Index 2+ predicts step-by-step servo movements.
            if logits.size(1) > 2:
                loss = F.mse_loss(logits[:, 2:, :], y[:, 2:, :])
            else:
                loss = F.mse_loss(logits, y)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_epoch_loss += batch_loss
            global_step += 1

            wandb.log({"step_loss": batch_loss, "global_step": global_step})

        avg_epoch_loss = total_epoch_loss / len(dataloader)
        wandb.log({"epoch": epoch, "epoch_loss": avg_epoch_loss})

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{config['epochs']} | Normalized MSE Loss: {avg_epoch_loss:.6f}")

    save_path = "robot_gpt.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")

    artifact = wandb.Artifact('robot-gpt-model', type='model')
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    train()

