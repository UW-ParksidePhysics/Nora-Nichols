l_t=[]
l_y=[]

a = 0
g = 9.81
n = eval(input("What is the number of iterations?"))
vo = eval(input("What is the initial velocity?"))
b = 2*vo/g
i = 0

for i in range (n+1):
    h = (b - a)/n
    t = a + i*h
    y = vo*t - (1/2)*g*t**2
    l_t.append(t)
    l_y.append(y)
    i += 1

for k in range (0,len(l_t),1):
    print (l_t[k],l_y[k])