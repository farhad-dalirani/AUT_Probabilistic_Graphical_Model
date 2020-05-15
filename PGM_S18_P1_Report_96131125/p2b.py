################################
#    Problem 2 - Subproblem b  #
################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from add_noise import add_noise
from simulated_annealing import simulated_anealing_multi_features
from PIL import Image

################################
#    read image and add noise  #
################################
# read noise free image
img_noise_free = Image.open('test2.jpg')

# convert to HSV
img_noise_free.convert('HSV')

width, height = img_noise_free.size
# resize image 33.33%, for reaching output faster
img_noise_free = img_noise_free.resize((int(width/3), int(height/3)), Image.ANTIALIAS)
img_noise_free = np.array(img_noise_free)

# convert it to Hue
# convert to range [0,1]
img_noise_free_Hue = np.double(img_noise_free[:, :, 0])/255

img_noise_free = Image.open('test2.jpg')
width, height = img_noise_free.size
# resize image 33.33%, for reaching output faster
img_noise_free = img_noise_free.resize((int(width/3), int(height/3)), Image.ANTIALIAS)
img_noise_free = np.array(img_noise_free)

# convert it to gray level (two dimensions matrix instead of three dimensions matrix)
# convert to range [0,1]
img_noise_free_gl = np.double(np.average(img_noise_free, weights=[0.299, 0.587, 0.114], axis=2))/255

# create new picture with features
image_multi_feature = np.zeros((img_noise_free_gl.shape[0], img_noise_free_gl.shape[1], 2))
image_multi_feature[:, :, 0] = img_noise_free_gl
image_multi_feature[:, :, 1] = img_noise_free_Hue
#print(image_multi_feature.shape)

############################################
#    Find Parameters For singleton clique  #
############################################
# prior of each class- equal priors is assumed
priors = {0: 1/3, 127: 1/3, 255: 1/3}

# three regions in noisy image that belongs to different classes.
# we use these three region for estimating mean and standard deviation
# of each class.
#regions = {0: (200, 1), 127: (10, 297), 255: (404, 394)}
#regions = {0: (160, 127), 127: (10, 195), 255: (264, 253)}
#regions = {0: (180, 500), 127: (0, 340), 255: (270, 320)}
regions = {0: (180, 500), 127: (0, 340), 255: (273, 350)}
len_of_region = 100

# calculate P(feature|class i), each of them it's an gaussian distribution
conditional_p = {0: None, 127: None, 255: None}
for key in conditional_p:
    # Find mean and standard deviation of P(features|class i)
    class_key_pixels_hue = img_noise_free_Hue[regions[key][0]:regions[key][0]+len_of_region,
                                regions[key][1]:regions[key][1] + len_of_region]

    # find intensity of all pixels of each class in the gaussian noisy image
    intensity_of_pixels_class_key_hue = class_key_pixels_hue.flatten()

    class_key_pixels_gl = img_noise_free_gl[regions[key][0]:regions[key][0] + len_of_region,
                       regions[key][1]:regions[key][1] + len_of_region]

    # find intensity of all pixels of each class in the gaussian noisy image
    intensity_of_pixels_class_key_gl = class_key_pixels_gl.flatten()

    intensity_of_pixels_class_key_hue = np.array(intensity_of_pixels_class_key_hue)
    intensity_of_pixels_class_key_gl = np.array(intensity_of_pixels_class_key_gl)
    features = np.vstack((intensity_of_pixels_class_key_gl, intensity_of_pixels_class_key_hue))

    # calculate mean and cov of intensities of pixels for each class
    mean_P_features_class_i = np.mean(features, axis=1)
    std_P_features_class_i = np.cov(features)
    conditional_p[key] = {'mean': mean_P_features_class_i, 'std': std_P_features_class_i}
print('>    P(feature|class i) of classes:\n', conditional_p)

#####################################################
#   Use Simulated annealing for optimization        #
#####################################################
print('\n>    Simulated Annealing is started, with 2000000 epochs, it takes several minutes, ...\n')
segmented_img = simulated_anealing_multi_features(img_multi_features=image_multi_feature, gaussians=conditional_p,
                                     neighbourhood='four', beta=0.1,
                                     initial_temperature=4000, labels_list=np.array([0, 127, 255]),
                                     max_iter=2000000)


##############################################################
#    Plot and save input image, noisy image and segmentation #
##############################################################
# plot images
plt.figure()
plt.imshow(img_noise_free_Hue, cmap='gray')
plt.title('Input Noise Free Image')
plt.figure()
plt.imshow(img_noise_free_Hue, cmap='gray')
plt.title('Input')
plt.figure()
plt.imshow(segmented_img, cmap='gray')
plt.title('Segmented image')

mpimg.imsave('output/p2b-1.png', img_noise_free_Hue, cmap='gray')
mpimg.imsave('output/p2b-2-segmented-image.png', segmented_img, cmap='gray')

plt.show()