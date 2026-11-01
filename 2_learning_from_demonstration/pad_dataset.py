# ============================================================================
# APPROACH 2: dataset clean-up
#
# Appends 40 copies of the final pose to the end of each demonstration, so the
# model learns to stop and hold the end position instead of drifting.
# Edits the files in place.
# ============================================================================

import glob
import os

# Directories containing your dataset files
DATA_DIRS = ["dataset/place", "dataset/reach2", "dataset/pick2"]
FILE_EXTENSIONS = ["*.csv", "*.txt"]
PADDING_FRAMES = 40  # Number of stationary end frames to append

def pad_file_in_place(file_path, num_padding=40):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"Skipping empty file: {file_path}")
        return

    # Grab the last line (the final target pose)
    last_line = lines[-1]
    if not last_line.endswith('\n'):
        last_line += '\n'

    # Append the last frame directly to the existing lines
    padded_lines = lines + [last_line] * num_padding

    # Overwrite the exact same file
    with open(file_path, 'w') as f:
        f.writelines(padded_lines)

    print(f"  ✓ Appended {num_padding} frames to {os.path.basename(file_path)} (Total: {len(padded_lines)} lines)")

if __name__ == "__main__":
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"Directory '{data_dir}' not found. Skipping...")
            continue

        print(f"\n--- Appending to files in: {data_dir} ---")
        
        files = []
        for ext in FILE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(data_dir, ext)))

        for file_path in files:
            pad_file_in_place(file_path, num_padding=PADDING_FRAMES)

    print("\nDataset updated directly in place! You're ready to retrain.")

