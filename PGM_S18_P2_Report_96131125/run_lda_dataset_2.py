from read_dataset_2 import read_dataset_2
from disply_samples import display_samples
from lda_by_sampling import lda_by_sampling
import matplotlib.pyplot as plt
import json
from titles_dataset2 import titles_dataset2

# read dataset 2
samples, word_num = read_dataset_2()
# print(dataset)
# print(dataset.shape)


#model 6
#W = word_num
#T = 20
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2


#model 7
#W = word_num
#T = 20
#alpha = 0.1
#beta = 1
#max_epoch = 120
#dataset = 2

#model 8
#W = word_num
#T = 20
#alpha = 1
#beta = 10
#max_epoch = 120
#dataset = 2

#model n
#W = word_num
#T = 10
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2

#model n+1
#W = word_num
#T = 20
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2

#model n,3
#W = word_num
#T = 30
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2

#model n,4
#W = word_num
#T = 50
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2

#model n,5
#W = word_num
#T = 2
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2

#model n,6
#W = word_num
#T = 70
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2

#model n+2
#W = word_num
#T = 100
#alpha = 1
#beta = 1
#max_epoch = 120
#dataset = 2


#model n,7
W = word_num
T = 30
alpha = 0.4
beta = 0.5
max_epoch = 120
dataset = 2


phi, theta, perplexities_epochs, total_time = lda_by_sampling(samples=samples, W=W, T=T,
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
with open('model-dataset2.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

# write titles in file (vocab form)
titles_dataset2(phi=phi)

# plot perplexity
plt.figure()
plt.plot(range(0, len(perplexities_epochs)), perplexities_epochs, label='Perplexity during training')
plt.legend()
plt.show()