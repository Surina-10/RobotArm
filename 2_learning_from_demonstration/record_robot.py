# ============================================================================
# APPROACH 2: recording demonstrations
#
# I move the real arm by hand through a task (pick, reach or place) while this
# script reads all 5 servo positions every 0.05 s. Each demonstration is saved
# as its own file in dataset/<task>/trajectory_N.txt:
#
#   line 1:   task header, a one-hot vector (pick = 0 0 0 0 1, reach = 0 0 0 1 0,
#             place = 0 0 1 0 0) that tells the model which task this is
#   line 2:   start token 999 999 999 999 999 (marks where motion begins)
#   line 3+:  one line per time step: the 5 servo positions
#
# These recordings are the training data for the transformer in gpt.py.
# ============================================================================

import os
import time
import lss
import lss_const as lssc

# ==========================================
# CONFIGURATION & HARDWARE SETUP
# ==========================================
CABLE_PORT = "/dev/ttyUSB0"
BAUD_RATE = lssc.LSS_DefaultBaud
SERVO_IDS = [1, 2, 3, 4, 5]

DATASET_DIR = "dataset"

# Configure tasks with custom 5-element header vectors
FOLDERS = {
    "1": ("pick", os.path.join(DATASET_DIR, "pick2"), [0, 0, 0, 0, 1]),
    "2": ("reach", os.path.join(DATASET_DIR, "reach2"), [0, 0, 0, 1, 0]),
    "3": ("place", os.path.join(DATASET_DIR, "place"), [0, 0, 1, 0, 0])
}

# Start token vector (5 elements)
START_TOKEN = [999, 999, 999, 999, 999]

# Ensure directories exist
for _, path, _ in FOLDERS.values():
    os.makedirs(path, exist_ok=True)

# Initialize Hardware Bus
try:
    lss.initBus(CABLE_PORT, BAUD_RATE)
    servos = {s_id: lss.LSS(s_id) for s_id in SERVO_IDS}
    print("Successfully connected to robot hardware.")
except Exception as e:
    print(f"Hardware connection failed: {e}")
    servos = None

# Read the 5 servo positions (LSS units: tenths of a degree).
def get_current_joint_states():
    """Reads current positions from Lynxmotion servos as integers."""
    if not servos:
        # Fallback dummy values if hardware isn't connected
        return [0] * len(SERVO_IDS)
       
    angles = []
    for s_id in SERVO_IDS:
        pos = servos[s_id].getPosition()
        # Round position and convert to whole integer
        angles.append(int(round(float(pos))) if pos is not None else 0)
    return angles

def get_folder_count(folder_path):
    """Returns total number of txt files in a directory."""
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.endswith('.txt')])

def display_folder_counts():
    """Displays only the file counts for each task directory."""
    print("\n--- Dataset Overview ---")
    for key, (label, path, _) in FOLDERS.items():
        count = get_folder_count(path)
        print(f"[{key}] Task '{label}': {count} files in folder")
    print("------------------------")

# ==========================================
# RECORDING LOGIC
# ==========================================
# Record one demonstration until Ctrl+C, then save it with header + start token.
def record_trajectory(target_file_path, task_header):
    print(f"\n[RECORDING STARTED] Saving to: {target_file_path}")
    print("Press Ctrl+C to stop recording...\n")
   
    trajectory_data = []
   
    try:
        while True:
            # 1. Fetch 5 joint angles (integers)
            joint_angles = get_current_joint_states()
            trajectory_data.append(joint_angles)
           
            # 2. Print servo positions on a NEW LINE each time (whole numbers)
            pos_str = " ".join([f"{val:4d}" for val in joint_angles])
            print(f"{pos_str}")
           
            time.sleep(0.05)
           
    except KeyboardInterrupt:
        print("\n\n[RECORDING STOPPED] Saving trajectory file...")

    # 3. Save to text file with Header & Start Token (space-separated integers)
    if trajectory_data:
        with open(target_file_path, "w") as f:
            # Line 1: Header array (e.g., 0 0 0 0 1)
            f.write(" ".join(str(int(val)) for val in task_header) + "\n")
           
            # Line 2: Start Token array (999 999 999 999 999)
            f.write(" ".join(str(int(val)) for val in START_TOKEN) + "\n")
           
            # Lines 3+: recorded 5-servo trajectories
            for step in trajectory_data:
                line = " ".join(str(int(val)) for val in step)
                f.write(f"{line}\n")
               
        print(f"[SUCCESS] Saved file with header, start token, and {len(trajectory_data)} frames.\n")
    else:
        print("[WARNING] No data was recorded.\n")

# ==========================================
# MAIN INTERACTIVE MENU
# ==========================================
# Menu: pick which task to record and the file number to save it as.
def main():
    while True:
        display_folder_counts()
       
        print("\nSelect Action:")
        print("1. Record 'pick' trajectory")
        print("2. Record 'reach' trajectory")
        print("3. Record 'place' trajectory")
        print("q. Quit")
       
        choice = input("\nEnter choice (1/2/3/q): ").strip().lower()
       
        if choice == 'q':
            print("Exiting recorder. Goodbye!")
            break
           
        if choice in FOLDERS:
            label, folder_path, task_header = FOLDERS[choice]
           
            print(f"\nTask selected: '{label}' (Header: {task_header})")
            num_choice = input("Enter trajectory number (e.g. 1 for 'trajectory_1.txt'): ").strip()
           
            if not num_choice.isdigit():
                print("[!] Invalid number. Returning to main menu.")
                continue
               
            file_name = f"trajectory_{num_choice}.txt"
            file_path = os.path.join(folder_path, file_name)
           
            # Warn if overwriting an existing file
            if os.path.exists(file_path):
                confirm = input(f"[!] WARNING: '{file_name}' already exists. Overwrite? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Action cancelled.")
                    continue
           
            record_trajectory(file_path, task_header)

if __name__ == "__main__":
    main()
