def function(a, b):
    """read a set of data points, find the quadratic of best fit, plot."""


import numpy as np
import matplotlib.pyplot as plt
import multipolyfit as mpf


m = 0.090
l = 0.089
g = 9.81


H = np.loadtxt("data.txt")
x, y = np.hsplit(H, 2)


plt.plot(x, y, 'kx')


stacked_x = np.array([x,x+1,x-1])
coeffs = mpf(stacked_x, y, deg)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
plt.plot(x2, y2, label="deg=3")