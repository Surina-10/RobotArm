# ============================================================================
# APPROACH 1 (final version): inverse kinematics with a Jacobian solver
#
# Goal: given a target point, calculate the servo positions that put the tip of
# the arm there, then send the servos to those exact numbers.
#
# How it works:
#   1. Base: rotate servo 1 to face the target, angle = atan2(y, x).
#   2. Arm: start from the servos' current positions and repeat up to 1000 times:
#        - forward kinematics -> where is the tip now?
#        - error = target - tip
#        - Jacobian transpose: each joint's column is the vector from that joint
#          to the tip, rotated 90 degrees; error . column says which way and
#          how much to turn that joint
#        - nudge every joint a small step (learning rate) in that direction
#      Stop when the tip is within epsilon of the target.
#   3. Check the answer with forward kinematics, then move servos 2-4.
#
# The commented-out section near the top is my earlier work on the way here:
# 2-link forward kinematics with trigonometry, the same thing with 4x4
# homogeneous transformation matrices, and a closed-form 2-link IK using the
# law of cosines.
#
# Started from Lynxmotion's example test.py (header below); the kinematics is
# mine. LSS units: positions are in tenths of a degree (900 = 90 deg).
# ============================================================================

###############################################################################
#	Author:			Sebastien Parent-Charette (support@robotshop.com)
#	Version:		1.0.0
#	Licence:		LGPL-3.0 (GNU Lesser General Public License version 3)
#	
#	Desscription:	An example using the LSS and the Python module.
###############################################################################

# Import required liraries
import time
import serial
import math

import numpy as np

# Import LSS library
import lss
import lss_const as lssc

# Constants
CST_LSS_Port = "/dev/ttyUSB0"		# For Linux/Unix platforms
#CST_LSS_Port = "COM230"				# For windows platforms
CST_LSS_Baud = lssc.LSS_DefaultBaud

# Create and open a serial port
lss.initBus(CST_LSS_Port, CST_LSS_Baud)

# Create LSS objects
myLSS1 = lss.LSS(1)
myLSS2 = lss.LSS(2)
myLSS3 = lss.LSS(3)
myLSS4 = lss.LSS(4)
myLSS5 = lss.LSS(5)


#myLSS1.move(300)
#myLSS2.move(300)
#myLSS3.move(300)
#myLSS4.move(300)
#myLSS5.move(300)

#print("myLSS1; position = " + str(myLSS1.getPosition()))
#print("myLSS2; position = " + str(myLSS2.getPosition()))
#print("myLSS3; position = " + str(myLSS3.getPosition()))
#print("myLSS4; position = " + str(myLSS4.getPosition()))
#print("myLSS5; position = " + str(myLSS5.getPosition()))
#print("myLSS1; model = " + str(myLSS1.getModel()))

#myLSS1.reset()
#myLSS2.reset()

#x0 = 0
#y0 = 0
#r1 = 6
#r2 = 6


# ---------------------------------------------------------------------
# EARLIER EXPLORATION (kept for reference, commented out)
# 1) 2-link forward kinematics with sin/cos
# 2) the same using 4x4 homogeneous transforms: Z = rotate about z,
#    X = translate along the link, so D = Z @ X @ A @ B gives the tip
# 3) closed-form 2-link inverse kinematics with the law of cosines
# ---------------------------------------------------------------------
# Arm reports angle of 0 degrees @ angle of 45 degrees from the horizontal.
# myLSS2.getPosition() returns angle in tenths of degrees
#theta1 = math.radians(float(myLSS2.getPosition())/10 - 45)
#theta2 = math.radians(float(myLSS3.getPosition())/10)


#x1 = x0 +r1*math.cos(theta1)
#y1 = y0 + r1*math.sin(theta1)
#x2 = x1 + r2*math.cos(theta2)
#y2 = y1 + r2*math.sin(theta2)

#print(theta1)
#print(theta2)
#print("")
#print(x1)
#print(y1)
#print("")
#print("End coordinates: (" + str(x2) + "," + str(y2) + ")")


#Z = np.array([[math.cos(theta1), -math.sin(theta1), 0, 0], [math.sin(theta1), math.cos(theta1), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#X = np.array([[1, 0, 0, r1], [0, math.cos(0), -math.sin(0), 0], [0, math.sin(0), math.cos(0), 0], [0, 0, 0, 1]])
#C = Z @ X

#A = np.array([[math.cos(theta2), -math.sin(theta2), 0, 0], [math.sin(theta2), math.cos(theta2), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#B = np.array([[1, 0, 0, r2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#D = Z @ X @ A @ B

#print("")
#print("X coordinate: ")
#print(D[0, 3])
#print("")
#print("Y coordinate: ")
#print(D[1, 3])
#print("")

#cos_theta2 = (x2*x2 + y2*y2 - r1*r1 - r2*r2)/(2*r1*r2)
#t2 = math.acos(cos_theta2)
#t1 = (math.atan2(y2, x2) - math.atan2(r2*math.sin(theta2), r1+r2*math.cos(theta2)))
#print("theta1: " + str(t1))
#print("theta2: " + str(t2))

# Destroy objects
#del myLSS1
#del myLSS2
#del myLSS3
#del myLSS4
#del myLSS5
# Destroy the bus
#lss.closeBus()

### EOF #######################################################################


#--------------------------------------------------------------------------------------------------------

# Configuration

# ---------------------------------------------------------------------
# FINAL VERSION: numerical inverse kinematics
# ---------------------------------------------------------------------
# Length of each link (shoulder->elbow, elbow->wrist, wrist->tip),
# in the same units as the target coordinates.
lengths = [6.0, 6.0, 4.0]

# Base Coordinates
x0 = 0
y0 = 0


# Forward Kinematics
# Forward kinematics: turn servo positions into the (x, y) of every joint.
# Each joint's angle adds onto the previous one, so the angles accumulate.
def forward_kinematics(lss_positions):
	# Create lists to hold coordinates for every joint
	# Starts at the base [x0, y0]
	joint_x = [x0]
	joint_y = [y0]
	accumulated_angle = 0.0

	# Loop though every joint 
	for i in range(len(lss_positions)):
		if i == 0:
			# LSS2 has a 45 degree base offset
			joint_deg = (float(lss_positions[i]) / 10.0) - 45.0
		else:
			# The other joints are relative to the previous arm
			joint_deg = float(lss_positions[i]) / 10.0

		accumulated_angle += math.radians(joint_deg)

		# Calculate next joint position based on cummulative vector angle
		next_x = joint_x[-1] + lengths[i] * math.cos(accumulated_angle)
		next_y = joint_y[-1] + lengths[i] * math.sin(accumulated_angle)

		joint_x.append(next_x)
		joint_y.append(next_y)

	# Returns a list of all X and a list of all Y coordinates
	# The final item in the lists is the end coordinate of the tip
	return joint_x, joint_y

# Jacobian Matrix & Gradient Loop
# Inverse kinematics by gradient steps with the Jacobian transpose.
# Returns the servo positions that put the tip on the target,
# or raises ValueError if it can't get there in 1000 iterations.
def inverse_kinematics_jacobain(target_radius, target_z, target_x, target_y, start_positions):
	# Copy starting pos into a list
	new_start_positions = [0.0 if pos is None else pos for pos in start_positions]
	current_lss = [float(pos) for pos in new_start_positions]
	num_joints = len(current_lss)

	# learning_rate: how big each nudge is; epsilon: how close counts as reached
	learning_rate = 0.001
	epsilon = 0.01

	for iteration in range(1000):
		# Track current posiitons across the entire arm
		jx, jy = forward_kinematics(current_lss)

		# End point is the last element in the coordinates list
		end_x = jx[-1]
		end_y = jy[-1]

		# Spatial Error Vector
		v_x = target_x - end_x
		v_y = (target_y - end_y)

		# Convergence Check
		if math.sqrt(v_x**2 + v_y**2) < epsilon:
			print("Jacobian converged succesfully in " + str(iteration) + " iterations")
			return current_lss

		# Build Jacobian  columns & multiply by error (Transpose Matrix Multiplication)
		# For each joint, the Jacobian entry is the perpendicular vector from that joint to the end tip
		for i in range(num_joints):
			# Calculate Jacobian column values for this specific joint index
			j_col_x = -(end_y - jy[i])
			j_col_y = (end_x - jx[i])

			# Matrix transpose multiplication to get the joint's gradient 
			d_theta_rad = (j_col_x * v_x) + (j_col_y * v_y)

			# Turn radian nudge into raw LSS values (*10) and update
			d_lss = math.degrees(d_theta_rad) * 10.0
			current_lss[i] += d_lss * learning_rate

		#myLSS2.move(int(current_lss[0]))
		#myLSS3.move(int(current_lss[1]))
		#time.sleep(0.2)

	raise ValueError("Jacobian loop timed out! Target is likely impossible to reach")

# Execution 
if __name__ == "__main__":
	# Define a target coordinate
	# The point I want the tip of the arm to reach.
	tx, ty = 5.0, 9.0
	tz = 5.0
	ty_new = -ty
	print("Target Goal: (" + str(tx) + "," + str(ty) + "," + str(tz) + ")")

	# Claw (servo 5) positions.
	CLAW_OPEN = -350
	CLAW_CLOSED = 0
	myLSS5.move(0)

	# STEP 1: rotate the base (servo 1) so the arm faces the target.
	base_angle_rad = math.atan2(ty_new, tx)
	base_angle_deg = math.degrees(base_angle_rad)

	base_target_lss = int(base_angle_deg * 10.0)
	myLSS1.move(base_target_lss)

	target_radius = math.sqrt(tx**2 + ty_new**2)

	# STEP 2: use the servos' current positions as the starting guess.
	live_lss2 = myLSS2.getPosition()
	live_lss3 = myLSS3.getPosition()
	live_lss4 = myLSS4.getPosition() 
	current_lss_positions = [live_lss2, live_lss3, live_lss4]

	# Put positions into array
	current_lss_positions = [live_lss2, live_lss3]

	try:
		# Run Calculation
		# STEP 3: solve for the joint positions.
		final_motor_values = inverse_kinematics_jacobain(target_radius, tz, tx, ty_new, current_lss_positions)

		for index, val in enumerate(final_motor_values):
			print("Target Raw Value for Joint Motor " + str(index + 1) + ": " + str(round(val, 1)))

		# Run verification check 
		# STEP 4: check the answer with forward kinematics, then move the servos.
		x_coords, y_coords = forward_kinematics(final_motor_values)
		print("Verified End Coordinate: (" + str(round(x_coords[-1], 2)) + "," + str(round(y_coords[-1], 2)) + ")")

		myLSS2.move(int(final_motor_values[0]))
		myLSS3.move(int(final_motor_values[1]))
		myLSS4.move(int(final_motor_values[-1]))

	except ValueError as error:
		print(error)
