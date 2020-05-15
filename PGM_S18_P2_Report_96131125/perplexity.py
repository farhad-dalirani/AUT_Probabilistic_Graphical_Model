def perplexity(samples, phi, theta):
    """
    Calculate perplexity
    :param samples:
    :param phi:
    :param theta:
    :return:
    """
    import numpy as np

    num_words = 0
    log_perplexity = 0

    # for all documents
    for i_doc, doc in enumerate(samples):
        # for all words
        for i_word in doc:
            log_perplexity += len(doc[i_word]) * np.log(np.inner(phi[:, i_word], theta[i_doc,:]))
            #print(np.inner(phi[:, i_word], theta[i_doc,:]))
            #print(np.log(np.inner(phi[:, i_word], theta[i_doc,:])))
            #num_words += len(doc[i_word])
        num_words += len(doc)

    perplexity_val = np.exp(-log_perplexity/num_words)

    return perplexity_val

