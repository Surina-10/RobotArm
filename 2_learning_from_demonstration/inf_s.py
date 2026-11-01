# ============================================================================
# APPROACH 2: running the per-task models on the real arm (improved version)
#
# Improvements over inf.py, added after testing on the real robot:
#   - seeds the model with 3 real recorded frames for the task instead of
#     a bare start token, so it starts from a realistic pose
#   - clamps every prediction to the servos' range, with a tighter range for
#     the claw so it doesn't stall (the servo's red stall light)
#   - smooths the output with an exponential moving average (alpha) so the
#     arm doesn't jitter
#   - after the motion, glides back to the home pose over 25 steps
# ============================================================================

import time
import os
import math
import torch
import lss
import lss_const as lssc
from gpt import RobotGPT

CABLE_PORT = "/dev/ttyUSB0"
BAUD_RATE = lssc.LSS_DefaultBaud
SERVO_IDS = [1, 2, 3, 4, 5]

# LSS Raw Units Range (-180.0 deg to +180.0 deg in tenths of a degree)
SERVO_MIN = -1800.0
SERVO_MAX = 1800.0
SERVO_RANGE = 3600.0  

# Mechanical safe bounds for Claw (Servo 5) to prevent red stall light
CLAW_MIN = -800.0
CLAW_MAX = 40.0

# Task-specific initial seed contexts
# Starting frames taken from real demonstrations, one set per task.
REACH_SEED_RAW = [
    [-959, 407, 465, 392, 10],
    [-959, 407, 467, 392, 10],
    [-959, 407, 468, 392, 10]
]

PICK_SEED_RAW = [
    [-966, 442, 449, 384, 6],
    [-966, 442, 449, 384, 6],
    [-966, 443, 447, 384, 6]
]

PLACE_SEED_RAW = [
    [-959, 407, 465, 392, 10],
    [-959, 407, 466, 392, 10],
    [-959, 407, 466, 391, 10]
]
def load_task_model(task_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RobotGPT(input_dim=5, d_model=128, n_head=4, n_layer=4, max_seq_len=200).to(device)
    
    model_filename = f"model_{task_name}.pt"
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file '{model_filename}' not found!")
        
    model.load_state_dict(torch.load(model_filename, map_location=device))
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

# Generate the motion step by step, clean it up, and send it to the servos.
def run_task_trajectory(task_name="pick", max_steps=120, alpha=0.15, RAMP_FRAMES=12):
    model, device = load_task_model(task_name)
    servos = connect_servos()

    raw_seed = PICK_SEED_RAW if task_name == "pick" else REACH_SEED_RAW
    norm_seed = [[(val - SERVO_MIN) / SERVO_RANGE for val in frame] for frame in raw_seed]
    context = torch.tensor([norm_seed], dtype=torch.float32, device=device)

    print(f"\n--- Executing Task: '{task_name.upper()}' ---")
    prev_angles = list(raw_seed[-1])

    with torch.no_grad():
        for step in range(max_steps):
            out = model(context)
            logits = out[0] if isinstance(out, tuple) else out
            
            next_norm_angles = logits[:, -1, :] 
            raw_angles = [(val.item() * SERVO_RANGE) + SERVO_MIN for val in next_norm_angles[0]]
            
            # Keep predictions inside the safe servo range.
            # 1. General arm clamping
            clamped_angles = [max(SERVO_MIN, min(SERVO_MAX, a)) for a in raw_angles]
            
            # 2. Specific Claw Protection (Servo 5 Clamp)
            clamped_angles[4] = max(CLAW_MIN, min(CLAW_MAX, clamped_angles[4]))

            if step < RAMP_FRAMES:
                ramp_factor = (1.0 - math.cos(math.pi * (step / RAMP_FRAMES))) / 2.0
                step_alpha = alpha * max(0.1, ramp_factor)
            elif step > (max_steps - RAMP_FRAMES):
                remaining = max_steps - step
                ramp_factor = (1.0 - math.cos(math.pi * (remaining / RAMP_FRAMES))) / 2.0
                step_alpha = alpha * max(0.1, ramp_factor)
            else:
                step_alpha = alpha
 
            # Blend each new prediction with the previous position to smooth the motion.
            # 3. EMA Filtering
            smoothed_angles = [
                alpha * curr + (1.0 - alpha) * prev 
                for curr, prev in zip(clamped_angles, prev_angles)
            ]
            prev_angles = smoothed_angles

            final_int_angles = [int(round(a)) for a in smoothed_angles]
            print(f"Step {step + 1:3d} -> Servos: {final_int_angles}")

            if servos:
                for idx, s_id in enumerate(SERVO_IDS):
                    servos[s_id].move(final_int_angles[idx])

            norm_step = torch.tensor(
                [[(a - SERVO_MIN) / SERVO_RANGE for a in smoothed_angles]], 
                dtype=torch.float32, 
                device=device
            ).unsqueeze(0)
            
            context = torch.cat((context, norm_step), dim=1)

            if context.size(1) > 150:
                context = context[:, -150:, :]

            time.sleep(0.05)

    # Return To Home Interpolation Phase
    # Return smoothly to where the motion started.
    print("\n--- Sequence finished. Transitioning back to Home Pose ---")
    target_home = raw_seed[0]
    for i in range(1, 26):
        interp_angles = [
            int(round(prev_angles[j] + (target_home[j] - prev_angles[j]) * (i / 25.0)))
            for j in range(5)
        ]
        if servos:
            for idx, s_id in enumerate(SERVO_IDS):
                servos[s_id].move(interp_angles[idx])
        time.sleep(0.05)

if __name__ == "__main__":
    task = input("Enter task ('pick' or 'reach or 'place'): ").strip().lower()
    run_task_trajectory(task_name=task, max_steps=100, alpha=0.15)

