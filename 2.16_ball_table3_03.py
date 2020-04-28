l_t=[]
l_y=[]
a = 0

g = 9.81

n = eval(input("What is the number of n"))

vo = eval(input("What is the Vo"))

b = 2*vo/g
i = 0

for i in range (n+1):
    h = (b - a)/n
    t = a + i*h
    y = vo*t - (1/2)*g*t**2
    l_t.append(t)
    l_y.append(y)
    i += 1
for y, t in zip(l_y,l_t):
    print (y, t)