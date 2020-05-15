def titles_dataset2(phi, threshold=0.08):
    """
    replace index word in theta by actual words and sort them according probabilities.
    :param phi:
    :return:
    """

    # read vocab
    with open('dataset2\\vocab.txt') as f:
        lines = f.readlines()

    vocab = []
    for line in lines:
        vocab.append(line.split()[0])
    #print(vocab)
    #print(len(vocab))

    theta_vocab = []
    for i_row in range(0, phi.shape[0]):
        theta_topic_i = phi[i_row, :].tolist()
        topic_i_vocab = [(x,y) for y, x in sorted(zip(theta_topic_i, vocab), reverse=True)]
        topic_i_vocab_important = []
        # keep important words for each topic
        first_prop = topic_i_vocab[0][1]
        for tuple_i in range(0, len(topic_i_vocab)):
            if topic_i_vocab[tuple_i][1] >= threshold*first_prop:
                topic_i_vocab_important.append(topic_i_vocab[tuple_i][0])
            else:
                break
        theta_vocab.append(topic_i_vocab_important)

    # write in file
    thefile = open('topics.txt', 'w')
    for item in theta_vocab:
        thefile.write("%s\n" % item)

    return theta_vocab


if __name__ == '__main__':
    #titles_dataset2(theta=None)
    pass