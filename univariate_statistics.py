def function(a, b):
    """Take a matrix and return a shape on a graph."""

import numpy as np
import networkx as nx
import pylab as plt


A = np.array([[0,0,1,0],[1,0,0,0],[1,0,0,1],[1,0,0,0]])   # This is a sample matrix. I couldn't figure out the Matlab one in Python.
G = nx.DiGraph(A)


pos = [[0,0], [0,1], [1,0], [1,1]]
nx.draw(G,pos)
plt.show()
