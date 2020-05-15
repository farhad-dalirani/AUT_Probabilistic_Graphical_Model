from read_dataset_1 import read_dataset_1, convert_dataset1
from disply_samples import display_samples
from lda_by_sampling import lda_by_sampling
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

# model1
#W = 25
#T = 10
#alpha = 1
#beta = 1
#max_epoch = 100
#dataset = 1

# model2
#W = 25
#T = 10
#alpha = 0.1
#beta = 1
#max_epoch = 120
#dataset = 1

# model3
#W = 25
#T = 10
#alpha = 10
#beta = 1
#max_epoch = 120
#dataset = 1


# model4
#W = 25
#T = 10
#alpha = 1
#beta = 0.1
#max_epoch = 160
#dataset = 1

# model5
#W = 25
#T = 10
#alpha = 1
#beta = 10
#max_epoch = 160
#dataset = 1


# model9
#W = 25
#T = 2
#alpha = 0.5
#beta = 0.5
#max_epoch = 160
#dataset = 1

# model10
#W = 25
#T = 5
#alpha = 0.5
#beta = 0.5
#max_epoch = 160
#dataset = 1

# model11
#W = 25
#T = 7
#alpha = 0.5
#beta = 0.5
#max_epoch = 160
#dataset = 1

# model12
#W = 25
#T = 10
#alpha = 0.5
#beta = 0.5
#max_epoch = 160
#dataset = 1

# model13
#W = 25
#T = 15
#alpha = 0.5
#beta = 0.5
#max_epoch = 160
#dataset = 1

# model14
#W = 25
#T = 20
#alpha = 0.5
#beta = 0.5
#max_epoch = 160
#dataset = 1

# model24
W = 25
T = 10
alpha = 0.5
beta = 0.5
max_epoch = 160
dataset = 2

phi, theta, perplexities_epochs, total_time = lda_by_sampling(samples=proper_dataset1, W=W, T=T,
                                                              alpha=alpha, beta=beta, max_epoch=max_epoch)

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
