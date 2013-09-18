#!/usr/bin/env python
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

### parse args
assert len(sys.argv) == 2, "Must provide a file to view"
filename = str(sys.argv[1])

### load file
f = open(filename, 'r')
data = pickle.load(f)
f.close()

### plot results
dims = data['dims']
runtimes = np.asarray(data['runtimes'])
simnames = data['sim_class_names']

plt.figure(1)
for i, name in enumerate(simnames):
    plt.plot(dims, runtimes[:,i], '.', markersize=30, label=name)

plt.legend(loc=2)
plt.show()


