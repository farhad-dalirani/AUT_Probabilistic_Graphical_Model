import numpy as np

def read_dataset_1():
    """
    read dataset that is introduced
    in 'Finding scientific topics Thomas L. Griffiths  and Mark Steyvers'.
    dataset is in form 2000*5*5 matrix. it's  2000 samples, each sample is
    5*5 matrix. sample i is 25 words and value of each in 5*5 matrix represents frequency of each word
    in sample i.
    :return: a numpy array with shape (2000, 25)
    """
    import numpy as np
    from scipy.io import loadmat
    dataset1 = loadmat('dataset1.mat')['fmaps_s']
    #print(dataset1.shape)
    for i_doc in range(0, dataset1.shape[0]):
        dataset1[i_doc, :, :] = np.transpose(dataset1[i_doc, :, :])

    # reshape images to 1-d array
    dataset = np.reshape(dataset1, newshape=(2000,-1))

    #dataset = np.array([[1, 4, 0, 3], [0, 2, 1, 1], [1, 1, 1, 1], [2, 3, 3, 1]])

    return dataset


def convert_dataset1(dataset1):
    """
    convert dataset1 to below proper form
    :param dataset1:  a numpy array with shape (number of samples, 25)
    :return: samples: a list that each element is a dictionary {i'th word: [1'th occurrence topic, ...,2'th occurrence topic]}
    """
    dataset = [{} for i in range(dataset1.shape[0])]
    for i in range(dataset1.shape[0]):
        for j in range(dataset1.shape[1]):
            if dataset1[i, j] != 0:
                dataset[i][j] = [-1 for k in range(dataset1[i, j])]

    return dataset


############################################
#             TEST UNIT
############################################
if __name__ == '__main__':
    import numpy as np
    dataset = read_dataset_1()
    print(dataset)
    #print(dataset.shape)
    print(np.reshape(dataset, newshape=(2000,5,-1))[0,:,:])
    print(dataset[0, :])

    proper_dataset = convert_dataset1(dataset1=dataset)
    print(proper_dataset[0])

    print(proper_dataset[0])
