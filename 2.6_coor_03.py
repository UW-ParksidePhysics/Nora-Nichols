a = 0
b = 10
n = 10

h = (b-a)/n

list = []
for i in range(0, n+1):
    list.append(a+i*h)
print(list)

c = [a+i*h for i in range(n+1)]
print(c)
