################################
#    Problem 1 - Subproblem b  #
################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from add_noise import add_noise

################################
#    read image and add noise  #
################################
# read noise free image
img_noise_free = mpimg.imread('test1.bmp')
# convert it to gray level (two dimensions matrix instead of three dimensions matrix)
# convert to range [0,1]
img_noise_free_gl = np.double(np.average(img_noise_free, weights=[0.299, 0.587, 0.114], axis=2))/255

for variance in [0.002, 0.009, 0.04, 0.08]:

    print('Variance: {}'.format(variance))

    # add gaussian noise to image
    img_gaussian_noise = add_noise(img=img_noise_free_gl, mean=0, variance=variance)

    ######################################
    #    Find Parameters of Naive Bayes  #
    ######################################
    # prior of each class- equal priors is assumed
    priors = {0: 1 / 3, 127: 1 / 3, 255: 1 / 3}

    # three regions in noisy image that belongs to different classes.
    # we use these three region for estimating mean and standard deviation
    # of each class.
    regions = {0: (15, 15), 127: (120, 15), 255: (171, 149)}
    len_of_region = 50

    # calculate P(feature|class i), each of them it's an gaussian distribution
    conditional_p = {0: None, 127: None, 255: None}
    for key in conditional_p:
        # Find mean and standard deviation of P(feature|class i)
        class_key_pixels = img_gaussian_noise[regions[key][0]:regions[key][0] + len_of_region,
                           regions[key][1]:regions[key][1] + len_of_region]

        # find intensity of all pixels of each class in the gaussian noisy image
        intensity_of_pixels_class_key = class_key_pixels[:]

        # calculate mean and std of intensities of pixels for each class
        intensity_of_pixels_class_key = np.array(intensity_of_pixels_class_key, dtype=np.float64)
        mean_P_feature_class_i = np.mean(intensity_of_pixels_class_key)
        std_P_feature_class_i = np.std(intensity_of_pixels_class_key, dtype=np.float64)
        conditional_p[key] = {'mean': mean_P_feature_class_i, 'std': std_P_feature_class_i}
    print('>    P(feature|class i) of classes:\n', conditional_p)

    #####################################################
    #    Find Label Of Each Pixel With Naive Bayes      #
    #####################################################
    segments = np.zeros(img_gaussian_noise.shape)

    # iterate all pixels
    for i in range(0, img_gaussian_noise.shape[0]):
        for j in range(0, img_gaussian_noise.shape[1]):

            # calculate p(class i)*p(feature|class i) for all classes
            # Naive Bayes selects class_i which has largest value
            probabilities = {}
            for key in priors:
                probabilities[key] = priors[key] * ((1 / (np.sqrt(2 * np.pi) * conditional_p[key]['std'])) *
                                                    (np.exp(
                                                        ((img_gaussian_noise[i, j] - conditional_p[key]['mean']) ** 2) /
                                                        (-2 * conditional_p[key]['std'] ** 2))))
            # print(probabilities)

            # class that has highest probability
            proper_class = max(probabilities, key=probabilities.get)
            segments[i, j] = proper_class

    #from  calculate_potential import calculate_potential
    #print(calculate_potential(img_gaussian_noise, conditional_p, segments, 'four', 1))
    #####################################################
    #           Calculate Accuracy                      #
    #####################################################
    # calculate accuracy
    acc = np.where((segments == img_noise_free[:, :, 0]) == True, 1, 0)
    acc = sum(sum(acc)) / (segments.shape[0] * segments.shape[1])

    print('\n##########################################')
    print('# Accuracy: {}%'.format(acc * 100))
    print('##########################################')

    ##############################################################
    #    Plot and save input image, noisy image and segmentation #
    ##############################################################
    # plot images
    plt.figure()
    plt.imshow(img_noise_free_gl, cmap='gray')
    plt.title('Input Noise Free Image')
    plt.figure()
    plt.imshow(img_gaussian_noise, cmap='gray')
    plt.title('Input After Adding Gaussian Noise (0, {})'.format(variance))
    plt.figure()
    plt.imshow(segments, cmap='gray')
    plt.title('Segmented image-accuracy: {}%'.format(acc*100))

    mpimg.imsave('output/p1b-1-noisy-image-var{0}.png'.format(variance), img_gaussian_noise, cmap='gray')
    mpimg.imsave('output/p1b-2-segmented-image-var{0}.png'.format(variance), segments, cmap='gray')

plt.show()
