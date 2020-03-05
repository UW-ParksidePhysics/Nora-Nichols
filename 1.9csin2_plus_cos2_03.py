a = 3.3; b = 5.3
a2 = a**2
b2 = b**2
eq1_sum = a2 + 2*a*b + b2
eq2_sum = a2 - 2*a*b + b2
eq1_pow = (a + b)**2
eq2_pow = (a - b)**2

print ('First Equation:', eq1_sum, '=', eq1_pow)
print ('Second Equation:', round(eq2_sum), '=', round(eq2_pow))