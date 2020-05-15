import matplotlib.pyplot as plt
import numpy as np

def display_samples(samples, plot_rows, plot_cols, title='some randomly selected samples', rand_sample=True):
    """
    display plot_rows * plot_cols images from samples
    :param samples: 2000*25 np.array, each row is a sample, each sample consists of
                        25 words, samples[i,j]: frequency of j'th word in i'th sample
    :param plot_rows: number of rows in plot
    :param plot_cols:
    :return:
    """
    fig, axes = plt.subplots(plot_rows, plot_cols)
    fig.suptitle(title)
    # randomly select some images

    if rand_sample == True:
        sub_samples = np.random.randint(samples.shape[0], size=plot_rows*plot_cols)
        sub_samples = sub_samples.tolist()
    else:
        sub_samples = [i for i in range(0, plot_rows*plot_cols)]

    for idx, index_in_samples in enumerate(sub_samples):
        # Get subplot row
        i = idx // plot_cols
        # Get subplot column
        j = idx - i*plot_cols
        #print(i, j)
        #print(type(axes), axes.shape)
        sample = samples[index_in_samples, :]
        sample = np.reshape(sample, newshape=(5, 5))
        if len(axes.shape) >=2:
            axes[i, j].imshow(sample, cmap='gray')
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])
        else:
            axes[j].imshow(sample, cmap='gray')
            axes[j].set_yticklabels([])
            axes[j].set_xticklabels([])

    plt.subplots_adjust(wspace=1, hspace=1)
    #plt.show()


############################################
#             TEST UNIT
############################################
if __name__ == '__main__':
    from read_dataset_1 import read_dataset_1

    dataset = read_dataset_1()
    display_samples(samples=dataset, plot_rows=5, plot_cols=5)