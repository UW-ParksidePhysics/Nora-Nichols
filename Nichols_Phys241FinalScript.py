#### part one, plotting the data points and a curve of best fit

# splitting the filename to show the chemical symbol, the chrystal symmetry symbol, and the density functional exchange-correlation approximation acronym


txt = "Au.Fm-3m.GGA-PBEsol.volumes_energies.dat"

x = txt.split(".")

print (x)


#Chemsym = Au
#ChrysSym = Fm-3m
#Acronym = GGA-PBEsol



import numpy as np
import matplotlib.pyplot as plt
import multipolyfit as mpf


m = 0.090
l = 0.089
g = 9.81


H = np.loadtxt("Au.Fm-3m.GGA-PBEsol.volumes_energies.dat")
x, y = np.hsplit(H, 2)


plt.plot(x, y, 'kx')


stacked_x = np.array([x,x+1,x-1])
coeffs = mpf("stacked_x, y")
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)                 #Evaluates the polynomial for each x2 value
plt.plot(x2, y2, label="deg=3")

plt.show

# Building my matrix

def generate_matrix(minimum_x, maximum_x, number_of_dimensions, potential_name, potential_parameter):
    """    
    Generates an NxN Hamiltonian matrix for a one-dimensional potential on a spatial grid    
    :param minimum_x:               float :: left endpoint of the spatial grid
    :param maximum_x:               float :: right endpoint of the spatial grid    
    :param number_of_dimensions:    int :: N, number of dimensions of the matrix and number of grid points of grid    
    :param potential_name:          str :: name of potential to use ('harmonic', 'sinusoidal', 'square')    
    :param potential_parameter:     float :: single parameter to adjust potential (affects magnitude of potential)
    :return:                        NumPy array (N,N) :: Hamiltonian matrix createdfrom potential
    """
    import numpy as np

    hbar = 1.0
    mass = 1.0

    grid_spacing = (maximum_x - minimum_x)/(number_of_dimensions - 1)
    units_prefactor = hbar**2 / (2 * mass * grid_spacing**2)

    horizontal_grid = np.linspace(minimum_x, maximum_x, num=number_of_dimensions)

    if potential_name == 'harmonic':
        angular_frequency = potential_parameter * hbar / (mass * (maximum_x - minimum_x)**2)
        reduced_potential = 0.5 * mass * np.power(angular_frequency * horizontal_grid, 2) / units_prefactor
    elif potential_name == 'sinusoidal':
        well_width = (maximum_x - minimum_x)
        wave_vector = np.pi / well_width
        pre_factor = 5 * potential_parameter * hbar**2 / (2. * mass * well_width**2)
        reduced_potential = pre_factor * np.sin(wave_vector * horizontal_grid) / units_prefactor
    elif potential_name == 'square':
        well_width = (maximum_x - minimum_x) / 2.
        well_depth = potential_parameter * well_width
        number_of_well_points = int(number_of_dimensions / 2.)
        number_of_outside_points = int(number_of_dimensions / 4.)
        reduced_potential = horizontal_grid
        reduced_potential[0:number_of_outside_points] = well_depth
        reduced_potential[number_of_well_points+number_of_outside_points:] = well_depth
        reduced_potential /= units_prefactor
    else:
        horizontal_grid = np.linspace(minimum_x, maximum_x, num=number_of_dimensions)
        reduced_potential = 0. * horizontal_grid
        off_diagonal_terms_array = -1. * np.ones(number_of_dimensions-1)
        diagonal_terms_array = np.full(number_of_dimensions, 2) + reduced_potential
        matrix_one = np.diagflat(off_diagonal_terms_array, -1)
        matrix_two = np.diagflat(diagonal_terms_array)
        matrix_three = np.diagflat(off_diagonal_terms_array, 1)
        matrix_total = units_prefactor*(matrix_one + matrix_two + matrix_three)
        return matrix_total

# time to make a matrix into a graph I guess

import networkx as nx
import pylab as plt


A = matrix_total   # this way I don't have to keep writing the longer name
G = nx.DiGraph(A)


pos = [[0,0], [0,1], [1,0], [1,1]]
nx.draw(G,pos)
plt.show()







#### part two, plotting the eigenvectors of a matrix

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

# generate num random points in 3d
num = 5

xcod = np.array([1, 1, 1, 1, 1, 1])
ycod = np.array([1, 1, 4.5, 5., 6, 1])
zcod = np.array([1, -2, 0, 2, 3, 1])

# plotting in 3d?

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plotting all the points

ax.plot(xcod, ycod, zcod, 'x-')

# adding labels for vertices

for i in range(num):
    ax.text(xcod[i], ycod[i], zcod[i], '%d(%.2f,%.2f,%.2f)' %
            (i, xcod[i], ycod[i], zcod[i]))

# supposed centroid

centroid = np.array([np.mean(xcod), np.mean(ycod), np.mean(zcod)])
ax.scatter(centroid[0], centroid[1], centroid[2], marker='o', color='r')

# labelling the axes

ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

# getting a stack of all vertices, while removing last repeat vertex

cod = np.vstack(
    (np.delete(xcod, -1), np.delete(ycod, -1), np.delete(zcod, -1)))

# caculating covariance matrix
# normalize with N-1

covmat = np.cov(cod, ddof=0)

# computing eigenvalues and eigenvectors

eigval, eigvec = LA.eig(covmat)

# multiplying eigenvalue and eigenvector

for vec in eigvec.T:

    vec += centroid
    drawvec = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color='r')

    # adding the arrow to the plot

    ax.add_artist(drawvec)

# showing the graph that was plotted

plt.show()
