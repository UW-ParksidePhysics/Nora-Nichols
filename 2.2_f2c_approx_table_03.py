print ('F, C, c') #table heading
F = 0 # start value for C
dF = 10 #increment of C in loop
while F <= 100: #loop heading with condition
    C = (5.0/9)*(F - 32) # 1st statement inside loop
    c = (F - 30)/2 #approximate value of c
    print (F, C, c) # 2nd statement inside loop
    F = F + dF #3d statement inside loop
print ('F, C, c') # end of table line after loop)