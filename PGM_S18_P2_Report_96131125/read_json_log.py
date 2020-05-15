import numpy as np
import json
from titles_dataset2 import titles_dataset2


"""
For Editing and testing vocab-titles
"""

path = 'model-21-dataset2\\model-dataset2.json'
with open(path, 'r') as outfile:
    data = json.load(outfile)

phi = data['phi']

phi = np.array(phi)
print(phi)

# write titles in file (vocab form)
titles_dataset2(phi=phi, threshold=0.12)