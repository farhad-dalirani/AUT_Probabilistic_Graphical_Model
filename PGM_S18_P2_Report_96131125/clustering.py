import numpy as np
import json
from sklearn.cluster import KMeans
import numpy as np


# read a theta of model 25 with is created by dataset 2, use the calculated theta
path = 'model-25-dataset2\\model-dataset2.json'
with open(path, 'r') as outfile:
    data = json.load(outfile)

# read theta of model 25
theta = data['theta']
theta = np.array(theta)

print('Theta:\n {}'.format(theta))
print('Shape of Theta:\n {}'.format(theta.shape))


# use kmeans clustering
kmeans = KMeans(n_clusters=30, random_state=0).fit(theta)
clusters_of_each_doc = kmeans.predict(theta)

clusters_of_doc = {i:[] for i in range(30)}
for i_doc in range(0, theta.shape[0]):
    clusters_of_doc[clusters_of_each_doc[i_doc]].append(i_doc)

print('Cluster of each document:')
print(clusters_of_doc)


thefile = open('doc-cluster.txt', 'w')
for item in clusters_of_doc:
  thefile.write("%s\n" % clusters_of_doc[item])