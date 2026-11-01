# ============================================================================
# APPROACH 2: dataset clean-up
#
# Rewrites older recordings into the current format (task header line, start
# token line, then whole-number servo positions). Edits the files in place.
# ============================================================================

import os

# Set up paths and headers
DATASET_DIR = "dataset"
TASKS = {
    "pick": [0, 0, 0, 0, 1],
    "reach": [0, 0, 0, 1, 0]
}
START_TOKEN = [999, 999, 999, 999, 999]

def convert_dataset_files():
    for task_name, header in TASKS.items():
        folder_path = os.path.join(DATASET_DIR, task_name)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            clean_lines = []

            # 1. New Header Line
            clean_lines.append(" ".join(str(x) for x in header))

            # 2. New Start Token Line
            clean_lines.append(" ".join(str(x) for x in START_TOKEN))

            # 3. Process remaining lines (convert floats to rounded integers)
            for line in lines:
                # Skip any old header/token markers if present in the data
                if "<START>" in line or line.startswith("1.0 0.0"):
                    continue

                parts = line.split()
                # Ensure we pull only the 5 servo position values
                if len(parts) >= 5:
                    servo_ints = [str(int(round(float(x)))) for x in parts[-5:]]
                    clean_lines.append(" ".join(servo_ints))

            # Overwrite the original file with updated formatting
            with open(file_path, "w") as f:
                f.write("\n".join(clean_lines) + "\n")

            print(f"Updated: {file_path}")

if __name__ == "__main__":
    convert_dataset_files()
    print("Done! All existing text files updated successfully.")
