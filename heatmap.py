import numpy
import matplotlib.pylab as plt
from os import listdir
import sys

def getNextFilename(filename_base):
    plots_dir = 'plots'
    filename_int_length = 2

    heatmap_files = [file for file in listdir(plots_dir) if file.startswith(filename_base)]
    if not heatmap_files:
        next_int = 0
    else:
        heatmap_files.sort()
        next_int = int(heatmap_files[-1][-6:-4]) + 1

    next_int_str = str(next_int).zfill(filename_int_length)
    return plots_dir + '/' + filename_base + '-' + next_int_str + '.png'
    

matrix = numpy.loadtxt("matrix")
matrix_magnify=numpy.zeros((matrix.shape[0]*10,matrix.shape[1]))
for i in range(matrix.shape[0]):
    for j in range(10):
        matrix_magnify[i*10+j,:]=matrix[i,:]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(matrix_magnify, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

if (len(sys.argv) > 1):
    filename_base = sys.argv[1]
else:
    filename_base = 'heatmap'

heatmapFileName = getNextFilename(filename_base)
print('Creating heatmp image: ' + fi)
plt.savefig(heatmapFileName)
