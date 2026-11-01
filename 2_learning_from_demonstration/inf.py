# ============================================================================
# APPROACH 2: running the trained model on the real arm (first version)
#
# Gives the model the task header + start token, then lets it generate motion
# one step at a time: each predicted set of 5 servo positions is sent to the
# arm and fed back in as input for the next prediction (autoregressive
# generation, the same way a GPT writes text one word at a time).
# ============================================================================

import time
import torch
import lss
import lss_const as lssc
from gpt import RobotGPT

# Optional Hardware Setup
CABLE_PORT = "/dev/ttyUSB0"
BAUD_RATE = lssc.LSS_DefaultBaud
SERVO_IDS = [1, 2, 3, 4, 5]

TASK_HEADERS = {
    "pick": [0.0, 0.0, 0.0, 0.0, 1.0],
    "reach": [0.0, 0.0, 0.0, 1.0, 0.0]
}
START_TOKEN = [999.0, 999.0, 999.0, 999.0, 999.0]

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RobotGPT(input_dim=5, d_model=128, n_head=4, n_layer=4, max_seq_len=200).to(device)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("robot_gpt.pt"))
    else:
        model.load_state_dict(torch.load("robot_gpt.pt", map_location=torch.device('cpu')))
        
    model.eval()
    return model, device

def connect_servos():
    try:
        lss.initBus(CABLE_PORT, BAUD_RATE)
        servos = {s_id: lss.LSS(s_id) for s_id in SERVO_IDS}
        print("Connected to LSS hardware.")
        return servos
    except Exception as e:
        print(f"Hardware connection skipped/failed: {e}")
        return None

# Generate max_steps of motion for a task and play it on the arm.
def generate_trajectory(task_name="pick", max_steps=50):
    model, device = load_model()
    servos = connect_servos()

    header = TASK_HEADERS.get(task_name, TASK_HEADERS["pick"])
    
    # Initialize prompt sequence with Header [Line 1] and Start Token [Line 2]
    context = torch.tensor([header, START_TOKEN], dtype=torch.float32, device=device).unsqueeze(0)

    print(f"\n--- Generating Trajectory for Task: '{task_name}' ---")
    
    with torch.no_grad():
        for step in range(max_steps):
            # Predict next step
            logits, _ = model(context)
            # Take prediction from the last token
            next_angles = logits[:, -1, :] 
            
            # Convert prediction to rounded integers
            rounded_angles = [int(round(val.item())) for val in next_angles[0]]
            
            print(f"Step {step + 1:2d} -> Predicted Servos: {rounded_angles}")

            # Send positions to physical servos if connected
            if servos:
                for idx, s_id in enumerate(SERVO_IDS):
                    servos[s_id].move(rounded_angles[idx])

            # Append predicted position to context for autoregressive loop
            next_tensor = torch.tensor([rounded_angles], dtype=torch.float32, device=device).unsqueeze(0)
            context = torch.cat((context, next_tensor), dim=1)

            time.sleep(0.05)

if __name__ == "__main__":
    task = input("Enter task to run ('pick' or 'reach'): ").strip().lower()
    generate_trajectory(task_name=task, max_steps=60)


