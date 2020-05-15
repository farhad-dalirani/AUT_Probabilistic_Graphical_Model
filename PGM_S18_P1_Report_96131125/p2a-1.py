################################
#    Problem 2 - Subproblem a  #
################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from add_noise import add_noise
from simulated_annealing import simulated_anealing
from PIL import Image

################################
#    read image and add noise  #
################################
# read noise free image
img_noise_free = Image.open('test2.jpg')
width, height = img_noise_free.size
# resize image 33.33%, for reaching output faster
img_noise_free = img_noise_free.resize((int(width/3), int(height/3)), Image.ANTIALIAS)
img_noise_free = np.array(img_noise_free)

# convert it to gray level (two dimensions matrix instead of three dimensions matrix)
# convert to range [0,1]
img_noise_free_gl = np.double(np.average(img_noise_free, weights=[0.299, 0.587, 0.114], axis=2))/255


############################################
#    Find Parameters For singleton clique  #
############################################
# prior of each class- equal priors is assumed
priors = {0: 1/3, 127: 1/3, 255: 1/3}

# three regions in noisy image that belongs to different classes.
# we use these three region for estimating mean and standard deviation
# of each class.
#regions = {0: (200, 1), 127: (10, 297), 255: (404, 394)}
regions = {0: (180, 500), 127: (0, 340), 255: (270, 320)}
regions = {0: (180, 500), 127: (0, 340), 255: (273, 350)}
len_of_region = 100

# calculate P(feature|class i), each of them it's an gaussian distribution
conditional_p = {0: None, 127: None, 255: None}
for key in conditional_p:
    # Find mean and standard deviation of P(feature|class i)
    class_key_pixels = img_noise_free_gl[regions[key][0]:regions[key][0]+len_of_region,
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
#   Use Simulated annealing for optimization        #
#####################################################
print('\n>    Simulated Annealing is started, with 2000000 epochs, it takes several minutes, ...\n')
segmented_img = simulated_anealing(img=img_noise_free_gl, gaussians=conditional_p,
                                     neighbourhood='four', beta=0.1,
                                     initial_temperature=4000, labels_list=np.array([0, 127, 255]),
                                     max_iter=2000000)


##############################################################
#    Plot and save input image, noisy image and segmentation #
##############################################################
# plot images
plt.figure()
plt.imshow(img_noise_free_gl, cmap='gray')
plt.title('Input Noise Free Image')
plt.figure()
plt.imshow(img_noise_free_gl, cmap='gray')
plt.title('Input')
plt.figure()
plt.imshow(segmented_img, cmap='gray')
plt.title('Segmented image-GrayLevel')

mpimg.imsave('output/p2a-gray-level-1.png', img_noise_free_gl, cmap='gray')
mpimg.imsave('output/p2a-gray-level-2-segmented-image.png', segmented_img, cmap='gray')

plt.show()