def function(a, b):
    """Read the file and return a two-column list."""

f = open("data.txt")


for x in f:
    print(x)


import pylab

datalist = [ ( pylab.loadtxt("data.txt"), label ) for filename, label in list_of_files ]


for data, label in datalist:
    pylab.plot( data[:,0], data[:,1], label=label )


pylab.legend()
pylab.title("Title of Plot")
pylab.xlabel("X Axis Label")
pylab.ylabel("Y Axis Label")


#  I remember doing something like this in gnuplot during Phys 201 and 202. Seemed a lot simpler and more efficient.

