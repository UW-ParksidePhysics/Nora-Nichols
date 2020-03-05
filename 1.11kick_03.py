Cd = 0.2 #coefficient of drag
Q = 1.2 #kg/m**3 #density of air
a = 0.11 #meters
from math import *
A = pi*a**2 #area in m**2
g = 9.8 #acceleration due to gravity in m/s**2
m = 0.43 #kg of mass
V1 = 33.33 #m/s of hard kick
V2 = 2.78 #m/s of soft kick
Fd1 = 1/2*Cd*Q*A*V1**2 #Drag force of hard kick
Fd2 = 1/2*Cd*Q*A*V2**2 #Drag force of soft kick
Fg = m*g #force due to gravity
print (Fd1)
print (Fd2)
print (Fg)