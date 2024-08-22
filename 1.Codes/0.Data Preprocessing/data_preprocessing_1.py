import os
import xml.etree.ElementTree as ET
import json
import numpy as np
nodes_list_file = './preprocessed/nodes.json'
with open(nodes_list_file, 'r') as f:
    nodes = json.load(f)

xml_files = './directed-geant-uhlig-15min-over-4months-ALL/'
xmlns = '{http://sndlib.zib.de/network}'
for filename in os.listdir(xml_files):
    tree = ET.parse(xml_files + filename)
    root = list(tree.iter(xmlns+'demands'))
    mat = np.zeros((len(nodes), len(nodes)))
    output_file = './preprocessed/' + ('_'.join(filename.split('.')[-2].split('-')[-2:])) + '.csv'
    if len(root) > 0:
        for demand in root[0]:
            mat[nodes[demand.find(xmlns + 'source').text], nodes[demand.find(xmlns + 'target').text]] = float(demand.find(xmlns + 'demandValue').text)
    # print(mat)
    np.savetxt(output_file, mat, delimiter=',', fmt='%f')
