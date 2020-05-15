from read_dataset_1 import read_dataset_1, convert_dataset1
from disply_samples import display_samples
from lda_by_sampling_more_z import lda_by_sampling_more_z
import matplotlib.pyplot as plt
import json

# read dataset 1
dataset_1 = read_dataset_1()
# print(dataset)
# print(dataset.shape)
# print(np.reshape(dataset, newshape=(2000, 5, -1))[1999, :, :])

# convert dataset1 to proper form for using in lda by gibbs's sampling
proper_dataset1 = convert_dataset1(dataset1=dataset_1)
# print(proper_dataset1[1999])

# model23
W = 25
T = 10
alpha = 0.5
beta = 0.5
max_epoch = 160
dataset = 2
phi, theta, perplexities_epochs, total_time = lda_by_sampling_more_z(samples=proper_dataset1, W=W, T=T,
                                                              alpha=alpha, beta=beta, max_epoch=max_epoch,
                                                              back_toback_z=True)

# model24
#W = 25
#T = 10
#alpha = 0.5
#beta = 0.5
#max_epoch = 160
#dataset = 2
#phi, theta, perplexities_epochs, total_time = lda_by_sampling_more_z(samples=proper_dataset1, W=W, T=T,
#                                                              alpha=alpha, beta=beta, max_epoch=max_epoch,
#                                                              back_toback_z=False)


print('================================')
print('phi:{}\n{}'.format(phi.shape, phi))
print('theta:{}\n{}'.format(theta.shape, theta))
print('Total Time for {} iterations: {} s, average time each iteration: {} s.'.format(max_epoch, total_time, total_time/max_epoch))

# save results in a json file
data={}
data['total_time'] = total_time
data['each_epoch_time'] = total_time/max_epoch
data['W'] = W
data['T'] = T
data['alpha'] = alpha
data['beta'] = beta
data['dataset'] = dataset
data['phi'] = phi.tolist()
data['theta'] = theta.tolist()
data['perplexities_epochs'] = perplexities_epochs
with open('model-dataset1.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

# display a subset of doc-image
display_samples(samples=dataset_1, plot_rows=5, plot_cols=5)
# display titles
display_samples(samples=phi, plot_rows=2, plot_cols=5)
plt.figure()
plt.plot(range(0, len(perplexities_epochs)), perplexities_epochs, label='Perplexity during training')
plt.legend()
plt.show()
