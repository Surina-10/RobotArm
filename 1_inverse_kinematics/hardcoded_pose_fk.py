# ============================================================================
# APPROACH 1 (first step): hardcoded servo positions + forward kinematics
#
# My first way of moving the arm: I worked out servo positions by hand and sent
# each servo straight to a fixed number with move(). (The move() lines are
# commented out so the script can also be run just to read the arm's pose.)
#
# It then reads back every servo's position and uses 2-link forward kinematics,
# x = r1*cos(t1) + r2*cos(t2), y = r1*sin(t1) + r2*sin(t2),
# to compute where the tip of the arm should be.
#
# Started from Lynxmotion's example test.py (header below); the arm-specific
# code is mine. LSS units: positions are in tenths of a degree (900 = 90 deg).
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

# Hardcoded pose: send each servo to a position I worked out by hand.
#myLSS1.move(107)
#myLSS2.move(-94)
#myLSS3.move(-277)
#myLSS4.move(-74)
#myLSS5.move(-30)

# Read back where each servo actually is.
print("myLSS1; position = " + str(myLSS1.getPosition()))
print("myLSS2; position = " + str(myLSS2.getPosition()))
print("myLSS3; position = " + str(myLSS3.getPosition()))
print("myLSS4; position = " + str(myLSS4.getPosition()))
print("myLSS5; position = " + str(myLSS5.getPosition()))
print("myLSS1; model = " + str(myLSS1.getModel()))

#myLSS1.reset()
#myLSS2.reset()

# Forward kinematics for a 2-link planar arm.
# r1, r2 = link lengths; theta1, theta2 = joint angles; (x0, y0) = base.
x0 = 0
y0 = 0
r1 = 3
r2 = 2

theta1 = 20
theta2 = 40

x1 = x0 + r1*math.cos(theta1)
y1 = y0 + r1*math.sin(theta1)
x2 = x1 + r2*math.cos(theta2)
y2 = y1 + r2*math.sin(theta2) 

print("End point coordinates: " + "(" + str(x2) + "," +str(y2) + ")")



# Destroy objects
del myLSS1
del myLSS2
del myLSS3
del myLSS4
del myLSS5

# Destroy the bus
lss.closeBus()

### EOF #######################################################################
