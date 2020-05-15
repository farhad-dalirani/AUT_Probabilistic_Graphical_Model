def add_noise(img, mean, variance):
    """
    This Function gets an image then it adds gaussian
    noise N(mean, variance) to image
    :param img: and gray-scale image- pixels has value between [0,1]
    :param mean: mean of noise
    :param variance: variance of gaussian noise
    :return:
    """
    import numpy as np

    np.random.seed(0)
    if len(img.shape) == 2:
        channel = 1
        row, col = img.shape
    else:
        row, col, channel = img.shape

    # create noise
    sigma = np.sqrt(variance)
    gaussian_noise = np.random.normal(loc=mean,scale=sigma, size=(row, col, channel))
    if len(img.shape) == 2:
        gaussian_noise = gaussian_noise.reshape(row, col)
    else:
        gaussian_noise = gaussian_noise.reshape(row, col, channel)

    # add noise to image
    img_gaussian_noise = np.add(np.double(img), np.double(gaussian_noise))
    img_gaussian_noise = np.clip(img_gaussian_noise, 0.0, 1.0)

    # return img_gaussian_noise image
    return img_gaussian_noise
