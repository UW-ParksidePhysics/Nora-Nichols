M1 = 47 #g
M2 = 67 #g
p = 1.038 #g/cm**3
C = 3.7 #J/g/K
K = 5.4810**(-3) #W/cm/K
Tw = 100 #Temperature in C of boiling water
T0f = 4 #Original temp in C
T0r = 20
Ty = 70 #Celsius degrees
from math import *
t1f = M1**(2/3)*C*p**(1/3)/(K*pi**2*(4*pi/3)**(2/3))*log(0.76*((T0f - Tw)/(Ty - Tw)))
t1r = M1**(2/3)*C*p**(1/3)/(K*pi**2*(4*pi/3)**(2/3))*log(0.76*((T0r - Tw)/(Ty - Tw)))
t2f = M2**(2/3)*C*p**(1/3)/(K*pi**2*(4*pi/3)**(2/3))*log(0.76*((T0f - Tw)/(Ty - Tw)))
t2r = M2**(2/3)*C*p**(1/3)/(K*pi**2*(4*pi/3)**(2/3))*log(0.76*((T0r - Tw)/(Ty - Tw)))
print (t1f)
print (t1r)
print (t2f)
print (t2r)