def read_dataset_2():
    """
    read dataset 2
    :return:samples: a list that each element is a dictionary {i'th word: [1'th occurrence topic, ...,2'th occurrence topic]}
                    , number of different words
    """

    # number of words
    word_num = -1

    # read files
    with open('dataset2\\ap.dat') as f:
        lines = f.readlines()

    dataset = [{} for i in range(len(lines))]
    for line_i, line in enumerate(lines):
        # split string
        word_fre = line.split()

        for word_i_th in range(1, len(word_fre)):
            word, fre = word_fre[word_i_th].split(':')

            word_num = max(word_num, int(word))
            if int(fre) != 0:
                dataset[line_i][int(word)] = [-1]*int(fre)

    return dataset, word_num+1


############################################
#             TEST UNIT
############################################
if __name__ == '__main__':
    dataset, word_num = read_dataset_2()

    print('Number of words: ', word_num)
    print('Number of docs: ', len(dataset))
    print(dataset[2245])