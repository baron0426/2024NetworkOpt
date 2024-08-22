import xml.etree.ElementTree as ET
import json
xml_file = './directed-geant-uhlig-15min-over-4months-ALL/demandMatrix-geant-uhlig-15min-20050504-1500.xml'
xmlns = '{http://sndlib.zib.de/network}'
tree = ET.parse(xml_file)
root = list(tree.iter(xmlns+'nodes'))
nodes = {}
if len(root) > 0:
    for (k, node) in enumerate(root[0]):
        nodes[node.attrib['id']] = k

output_file = './preprocessed/nodes.json'
with open(output_file, 'w') as f:
    json.dump(nodes, f)
