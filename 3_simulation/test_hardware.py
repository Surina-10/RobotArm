# ============================================================================
# APPROACH 3: hardware check
#
# Opens the USB serial connection to the LSS servos and reads servo 1's
# position, to confirm the computer can talk to the arm.
# ============================================================================

import lss

# Update port name if necessary (/dev/ttyUSB0 or /dev/ttyACM0 on Linux, COMx on Windows)
PORT = "/dev/ttyUSB0" 
BAUD = 115200

try:
    lss.initBus(PORT, BAUD)
    print(f"Opened serial port: {PORT}")
    
    # Ping Servo ID 1
    s1 = lss.LSS(1)
    print("Servo 1 Position (tenths of deg):", s1.getPosition())
    
except Exception as e:
    print(f"Could not connect to servos: {e}")
