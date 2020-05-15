def lda_by_sampling(samples, W, T, alpha, beta, max_epoch):
    """
    lda by gibbs sampling
    :param samples: a list that each element is a dictionary {i'th word: [1'th occurrence topic, ...,2'th occurrence topic]}
    :param W: total number of words in all documents
    :param T: total number of topics
    :param alpha: parameter that sets the topic distribution
    :param beta: parameter that sets the topic distribution for the words
    :param max_epoch: how many iteration should be done to reach mixing
    :return:
    """
    import random as rand_py
    import numpy as np
    from perplexity import perplexity
    import time
    #np.random.seed(0)
    # number of documents
    D = len(samples)

    # initialize topic of each word of each document
    for i_doc in range(0, D):
        for word in samples[i_doc]:
            for word_j_occurrence in range(0, len(samples[i_doc][word])):
                samples[i_doc][word][word_j_occurrence] = rand_py.randint(0, T-1)
    #print(samples[1999])
    #print(samples)

    # word-topic matrix
    wt = np.zeros(shape=(T, W))
    for i_doc in range(0, D):
        for word in samples[i_doc]:
            for word_j_occurrence in range(0, len(samples[i_doc][word])):
                wt[samples[i_doc][word][word_j_occurrence]][word] += 1
    #print('wt')
    #print(wt)

    # document-topic matrix
    dt = np.zeros(shape=(D, T))
    for i_doc in range(0, D):
        for word in samples[i_doc]:
            for word_j_occurrence in range(0, len(samples[i_doc][word])):
                dt[i_doc][samples[i_doc][word][word_j_occurrence]] += 1
    #print('dt', D, T)
    #print(dt)

    # different topics
    topics = [i for i in range(0,T)]

    # perplexities in different epochs
    perplexities = []

    # total time
    time_total = 0

    # do sufficient iteration to reach mixing
    for i_iter in range(0, max_epoch):

        # time
        time_a = time.time()

        print('\r epoch {}/{} ... '.format(i_iter, max_epoch))
        wt_copy = np.copy(wt)
        dt_copy = np.copy(dt)

        if i_iter == 0:
            # total number of words in each topic.
            wt_j = np.sum(wt, axis=1)
            #print('shape wt_j {}'.format(wt_j.shape))
            # total number of words in document i.
            dt_di = np.sum(dt, axis=1)
            #print('shape dt_di {}'.format(dt_di.shape))

        # for all words in each document change topic by calculating and using
        # transition P(zi=j | Z-i, wi, di) for different j
        for i_doc in range(0, D):

            # transitions
            transitions = {word:[0]*T for word in samples[i_doc]}
            for word in transitions:
                for i_topic in range(0, T):
                    transitions[word][i_topic] = ((wt[i_topic, word]+beta)/(wt_j[i_topic]+W*beta)) *\
                                                 ((dt[i_doc, i_topic]+alpha)/(dt_di[i_doc]+T*alpha))
                # normalize
                normalize_fac = sum(transitions[word])
                transitions[word] = [transitions[word][i_topic]/normalize_fac for i_topic in range(0, T)]

            #print('Transition')
            #print(transitions)

            for word in samples[i_doc]:
                for word_j_occurrence in range(0, len(samples[i_doc][word])):
                    # select new topic for word
                    new_topic = np.random.choice(a=topics, p=transitions[word])

                    # update topic of word, dt and wt
                    if new_topic != samples[i_doc][word][word_j_occurrence]:
                        wt_copy[samples[i_doc][word][word_j_occurrence]][word] -= 1
                        wt_copy[new_topic][word] += 1
                        dt_copy[i_doc][samples[i_doc][word][word_j_occurrence]] -= 1
                        dt_copy[i_doc][new_topic] += 1
                        samples[i_doc][word][word_j_occurrence] = new_topic

        # update previous word-topic and document-topic
        wt = np.copy(wt_copy)
        dt = np.copy(dt_copy)

        # calculate theta and phi
        # total number of words in each topic.
        wt_j = np.sum(wt, axis=1)
        # print('shape wt_j {}'.format(wt_j.shape))
        # total number of words in document i.
        dt_di = np.sum(dt, axis=1)
        # print('shape dt_di {}'.format(dt_di.shape))


        time_b = time.time()
        time_total += time_b-time_a

        # calculate phi
        phi = np.zeros(shape=(T, W))
        for j_topic in range(0, T):
            for i_word in range(0, W):
                phi[j_topic, i_word] = ((wt[j_topic, i_word] + beta) / (wt_j[j_topic] + W * beta))

        # calculate theta
        theta = np.zeros(shape=(D, T))
        for d_doc in range(0, D):
            for j_topic in range(0, T):
                theta[d_doc, j_topic] = ((dt[d_doc, j_topic] + alpha) / (dt_di[d_doc] + T * alpha))

        perplexity_epoch_i = perplexity(samples=samples, phi=phi, theta=theta)
        perplexities.append(perplexity_epoch_i)

    return phi, theta, perplexities, time_total



############################################
#             TEST UNIT
############################################
if __name__ == '__main__':
    from read_dataset_1 import read_dataset_1, convert_dataset1
    from disply_samples import display_samples

    # read dataset 1
    #dataset = read_dataset_1()
    #print(dataset)
    #print(dataset.shape)
    #print(np.reshape(dataset, newshape=(2000, 5, -1))[1999, :, :])

    #convert dataset1 to proper form for using in lda by gibbs's sampling
    #proper_dataset1 = convert_dataset1(dataset1=dataset)
    #print(proper_dataset1[1999])

    #wt, dt = lda_by_sampling(samples=proper_dataset1, W=25, T=10, alpha=1, beta=1, max_epoch=100)
    
    #print('================================')
    #print('wt:{}\n{}'.format(wt.shape, wt))
    #print('dt:{}\n{}'.format(dt.shape, dt))

    #display_samples(samples=wt, plot_rows=2, plot_cols=5, title='word-topic matrix', rand_sample=False)
    pass