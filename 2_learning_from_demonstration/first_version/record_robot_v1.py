# ============================================================================
# APPROACH 2, first version of the recorder
#
# Puts every servo in limp mode so I can move the arm by hand, then reads all
# 5 servo positions every 0.1 s and saves them to a CSV until I press Ctrl+C.
# first_recording.csv in this folder is one of these recordings.
# ============================================================================

import time
import os
import lss
import lss_const as lssc

# 1. Configuration Setup
CABLE_PORT = "/dev/ttyUSB0"  
BAUD_RATE = lssc.LSS_DefaultBaud
data_log_path = "robot_positions.txt"

# Define your 5 servo IDs
servo_ids = [1, 2, 3, 4, 5]

# 2. Initialize the LSS Bus Connection
try:
    lss.initBus(CABLE_PORT, BAUD_RATE)
    print(f"Connected to robot on port {CABLE_PORT} successfully.")
except Exception as e:
    print(f"Error connecting to the robot: {e}")
    exit()

# --- CRITICAL FIX: Ensure 'servos' dictionary is initialized right here ---
servos = {}

print("\nInitializing servo objects and setting them to Limp Mode...")
for s_id in servo_ids:
    servos[s_id] = lss.LSS(s_id)
    servos[s_id].limp()

print("Ready! Move the robot around. Recording coordinates every 0.1 seconds.")
print("Press Ctrl+C to stop recording and save the file.\n")

# 4. Open the file and start recording positions
with open(data_log_path, "w", encoding="utf-8") as f:
    f.write("servo1,servo2,servo3,servo4,servo5\n")
   
    try:
        while True:
            current_positions = []
           
            for s_id in servo_ids:
                # This will now safely find 'servos' defined above
                pos = servos[s_id].getPosition()
               
                if pos is None:
                    pos = 0
               
                current_positions.append(str(pos))
           
            line_of_data = ",".join(current_positions)
            print(f"Recorded Step: {line_of_data}")
           
            f.write(line_of_data + "\n")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\nRecording stopped! All data safely saved to '{data_log_path}'.")
