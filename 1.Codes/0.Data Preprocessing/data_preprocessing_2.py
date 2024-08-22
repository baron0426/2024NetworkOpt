import os
import numpy as np
csv_files = './preprocessed/'
matrices = {}
for filename in os.listdir(csv_files):
    if not filename.endswith('csv'):
        continue
    date = filename.split('_')[0]
    mat = np.loadtxt(csv_files+filename, delimiter=',')
    if date in matrices:
        matrices[date] += mat
    else:
        matrices[date] = mat

output_files = './preprocessed_daily/'
for (date, matrix) in matrices.items():
    np.savetxt(output_files+date+'.csv', matrix, delimiter=',', fmt='%f')
