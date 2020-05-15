################################
#    Problem 1 - Subproblem c  #
################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from add_noise import add_noise
from simulated_annealing import simulated_anealing

################################
#    read image and add noise  #
################################
# read noise free image
img_noise_free = mpimg.imread('test1.bmp')
# convert it to gray level (two dimensions matrix instead of three dimensions matrix)
# convert to range [0,1]
img_noise_free_gl = np.double(np.average(img_noise_free, weights=[0.299, 0.587, 0.114], axis=2))/255

# add gaussian noise to image
variance = 0.05
img_gaussian_noise = add_noise(img=img_noise_free_gl, mean=0, variance=variance)

############################################
#    Find Parameters For singleton clique  #
############################################
# prior of each class- equal priors is assumed
priors = {0: 1/3, 127: 1/3, 255: 1/3}

# three regions in noisy image that belongs to different classes.
# we use these three region for estimating mean and standard deviation
# of each class.
regions = {0: (15, 15), 127: (120, 15), 255: (171, 149)}
len_of_region = 50

# calculate P(feature|class i), each of them it's an gaussian distribution
conditional_p = {0: None, 127: None, 255: None}
for key in conditional_p:
    # Find mean and standard deviation of P(feature|class i)
    class_key_pixels = img_gaussian_noise[regions[key][0]:regions[key][0]+len_of_region,
                                regions[key][1]:regions[key][1] + len_of_region]

    # find intensity of all pixels of each class in the gaussian noisy image
    intensity_of_pixels_class_key = class_key_pixels[:]

    # calculate mean and std of intensities of pixels for each class
    intensity_of_pixels_class_key = np.array(intensity_of_pixels_class_key, dtype=np.float64)
    mean_P_feature_class_i = np.mean(intensity_of_pixels_class_key)
    std_P_feature_class_i = np.std(intensity_of_pixels_class_key, dtype=np.float64)
    conditional_p[key] = {'mean': mean_P_feature_class_i, 'std': std_P_feature_class_i}
print('>    P(feature|class i) of classes:\n', conditional_p)

# for different T0 (initial temperature) do segmentation
for initial_temperature in [1, 100, 10000, 1000000, 100000000]:
    print('>    Initial Temperature: {}'.format(initial_temperature))
    #####################################################
    #   Use Simulated annealing for optimization        #
    #####################################################
    print('\n>    Simulated Annealing is started, with 2000000 epochs, it takes at most 2 minutes, ...\n')
    segmented_img = simulated_anealing(img=img_gaussian_noise, gaussians=conditional_p,
                                         neighbourhood='four',
                                         beta=1,
                                         initial_temperature=initial_temperature,
                                         labels_list=np.array([0, 127, 255]),
                                         max_iter=2000000)

    #####################################################
    #           Calculate Accuracy                      #
    #####################################################
    # calculate accuracy
    acc = np.where((segmented_img == img_noise_free[:, :, 0]) == True, 1, 0)
    acc = sum(sum(acc))/(segmented_img.shape[0]*segmented_img.shape[1])

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
    plt.imshow(segmented_img, cmap='gray')
    plt.title('Segmented image-T0-{}'.format(initial_temperature))

    mpimg.imsave('output/p1h-1-noisy-image-var-{}.png'.format(variance), img_gaussian_noise, cmap='gray')
    mpimg.imsave('output/p1h-2-segmented-image-T0-{}.png'.format(initial_temperature), segmented_img, cmap='gray')

# for different temperature coefficients which multiplied in temperature
# in each epoch to decrease temperature do segmentation
for tempreture_coef in [0.01, 0.1, 0.6, 0.95, 1]:
    print('>    Temperature Coefficient: {}'.format(tempreture_coef))
    #####################################################
    #   Use Simulated annealing for optimization        #
    #####################################################
    print('\n>    Simulated Annealing is started, with 2000000 epochs, it takes at most 2 minutes, ...\n')
    segmented_img = simulated_anealing(img=img_gaussian_noise, gaussians=conditional_p,
                                       neighbourhood='four',
                                       beta=1,
                                       initial_temperature=4000,
                                       labels_list=np.array([0, 127, 255]),
                                       max_iter=2000000,
                                       tempreture_coef=tempreture_coef)

    #####################################################
    #           Calculate Accuracy                      #
    #####################################################
    # calculate accuracy
    acc = np.where((segmented_img == img_noise_free[:, :, 0]) == True, 1, 0)
    acc = sum(sum(acc)) / (segmented_img.shape[0] * segmented_img.shape[1])

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
    plt.imshow(segmented_img, cmap='gray')
    plt.title('Segmented image-Temperature-Coefficient-{}'.format(tempreture_coef))

    mpimg.imsave('output/p1h-1-noisy-image-var-{}.png'.format(variance), img_gaussian_noise, cmap='gray')
    mpimg.imsave('output/p1h-2-segmented-image-Temp_coef-{}.png'.format(tempreture_coef), segmented_img, cmap='gray')

plt.show()

