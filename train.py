from Neuron import Neuron
from Lattice import Lattice

# args are row, col, numWeights, numIterations
network = Lattice(10, 10, 4, 1500)

network.train()
