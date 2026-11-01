# ============================================================================
# APPROACH 3: sim-to-real, running the policy on the physical arm
#
# Connects to the 4 arm servos, then loops: read the real joint angles,
# estimate joint velocities, ask the trained policy for the next move, and
# send it to the servos. Each move is capped (MAX_DELTA) for safety, and once
# the arm is close to the goal it just holds the goal position.
# ============================================================================

import time
import numpy as np
from stable_baselines3 import PPO

# Official Lynxmotion LSS Python Driver
import lss
import lss_const as lss_c

# Serial Port Configuration
C_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

SERVO_IDS = [1, 2, 3, 4]
MAX_DELTA = 0.015


def setup_servos():
    """Initialize serial connection to Lynxmotion Smart Servos."""
    lss.initBus(C_PORT, BAUD_RATE)
    for s_id in SERVO_IDS:
        lss.LSS(s_id).setAngularHoldingStiffness(-2)
        lss.LSS(s_id).setMaxSpeed(300)


def get_real_joint_positions():
    """Query current positions from real LSS servos in radians cleanly without terminal clutter."""
    positions = []
    for s_id in SERVO_IDS:
        pos = lss.LSS(s_id).getPosition()

        if pos is not None and pos != "":
            try:
                deg = float(pos) / 10.0
            except ValueError:
                deg = 0.0
        else:
            deg = 0.0

        rad = np.radians(deg)
        positions.append(rad)

    return np.array(positions, dtype=np.float32)


def send_joint_commands(target_angles_rad):
    """Send target joint positions in radians to physical servos."""
    for s_id, target_rad in zip(SERVO_IDS, target_angles_rad):
        target_deg_tenths = int(np.degrees(target_rad) * 10)
        lss.LSS(s_id).move(target_deg_tenths)


def main():
    setup_servos()

    model = PPO.load("lss_arm_policy.zip")

    # Fixed target goal matching train.py
    target_goal = np.array([0.3, 0.4, -0.2, 0.1], dtype=np.float32)
    dt = 0.05
    prev_joints = get_real_joint_positions()

    try:
        while True:
            # 1. Read current joint angles
            current_joints = get_real_joint_positions()

            # 2. Estimate joint velocities
            joint_velocities = np.clip((current_joints - prev_joints) / dt, -2.0, 2.0)
            prev_joints = current_joints.copy()

            # 3. Compute distance to target goal
            dist_to_goal = np.linalg.norm(target_goal - current_joints)

            # 4. Goal Reached Check: Hold target goal directly once arrived
            if dist_to_goal < 0.03:
                send_joint_commands(target_goal)
                time.sleep(dt)
                continue

            # 5. Create exact 9-element observation vector
            obs = np.concatenate([current_joints, joint_velocities, [dist_to_goal]]).astype(np.float32)

            # 6. Predict step action
            action, _states = model.predict(obs, deterministic=True)

            # 7. Calculate incremental target pose
            target_cmd = current_joints + (action * MAX_DELTA)

            # 8. Send command to physical arm
            send_joint_commands(target_cmd)

            time.sleep(dt)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

