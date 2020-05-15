def calculate_potential(img, gaussians, class_map, neighbourhood, beta):
    """
    This function gets an image, p(feature|class i) and predicted label
    of each pixel, then it calculates potential of labeled image
    :param img: input image, range of grayscale is in range [0, 1]
    :param gaussians: p(feature|class i), which is a gaussian distribution
    :param class_map: predicted label for each pixel
    :param neighbourhood: if it be 'four', it will consider 4 of its neighbours,
                        if it be 'eight', it will consider 8 of neighbours.
    :param beta: weight of neighbours
    :return: return a scalar which is potential of labeled image
    """
    import numpy as np

    # potential of labeled image
    u_potential = np.float64(0)
    # potential for singleton
    u_singleton = np.float64(0)
    # potential for doubleton
    u_doubleton = np.float64(0)

    # iterate all pixels
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):

            # calculate singleton potential for pixel i,j
            u_singleton += np.log(np.sqrt(2*np.pi)*gaussians[class_map[i, j]]['std']) +\
                           ((img[i, j]-gaussians[class_map[i, j]]['mean'])**2)/(2*(gaussians[class_map[i,j]]['std']**2))
            # print(u_singleton)

            # calculate doubleton potential for pixel i,j and its neighbours
            if neighbourhood == 'four' or neighbourhood == 'eight':
                if i-1 >= 0:
                    if class_map[i,j] == class_map[i-1, j]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i+1 < img.shape[0]:
                    if class_map[i, j] == class_map[i+1, j]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if j-1 >= 0:
                    if class_map[i,j] == class_map[i, j-1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if j+1 < img.shape[1]:
                    if class_map[i, j] == class_map[i, j+1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
            else:
                raise ValueError('neighbourhood parameter is wrong!')

            if neighbourhood == 'eight':
                if i-1 >= 0 and j-1 >= 0:
                    if class_map[i,j] == class_map[i-1, j-1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i-1 >= 0 and j + 1 < img.shape[1]:
                    if class_map[i, j] == class_map[i-1, j+1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i+1 < img.shape[0] and j-1 >= 0:
                    if class_map[i,j] == class_map[i+1,j-1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i+1 < img.shape[0] and j+1 < img.shape[1]:
                    if class_map[i, j] == class_map[i+1, j+1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta

    # potential of input labeled image
    u_potential = u_singleton + u_doubleton

    # return potential
    return u_potential


def calculate_potential_by_modification(img, gaussians, class_map, neighbourhood, beta, index_ij, new_label, potential):
    """
    This function doesn't calculate potential from scratch, it calculates
    potential of new stated which obtained by modifying label of one pixel in effective
    way. it calculates difference of changing label of specific pixel and updates
    potential based on potential before changing label of pixel.
    :param img: input image, range of grayscale is in range [0, 1]
    :param gaussians: p(feature|class i), which is a gaussian distribution
    :param class_map: predicted label for each pixel
    :param neighbourhood: if it be 'four', it will consider 4 of its neighbours,
                        if it be 'eight', it will consider 8 of neighbours.
    :param beta: weight of neighbours
    :param index_ij: pixel which its label is proposed to change
    :param new_label: new label of pixel_ij
    :param potential: potential of labeled image before changing the label of pixel_ij
    :return: return a scalar which is potential of new labeled image
    """
    import numpy as np

    u_potential = potential

    i = index_ij[0]
    j = index_ij[1]

    # subtract singleton potential of pixel index_ij
    u_potential -= np.log(np.sqrt(2*np.pi)*gaussians[class_map[i, j]]['std']) +\
                           ((img[i, j]-gaussians[class_map[i, j]]['mean'])**2)/\
                           (2*(gaussians[class_map[i, j]]['std']**2))

    # add singleton potential of pixel index_ij with new label
    u_potential += np.log(np.sqrt(2 * np.pi) * gaussians[new_label]['std']) + \
                   ((img[index_ij[0], index_ij[1]] - gaussians[new_label]['mean']) ** 2) / \
                   (2 * (gaussians[new_label]['std'] ** 2))

    i = index_ij[0]
    j = index_ij[1]
    # subtract doubleton potential for pixel index_ij and its neighbours
    if neighbourhood == 'four' or neighbourhood == 'eight':
        if i - 1 >= 0:
            if class_map[i, j] == class_map[i - 1, j]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i + 1 < img.shape[0]:
            if class_map[i, j] == class_map[i + 1, j]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if j - 1 >= 0:
            if class_map[i, j] == class_map[i, j - 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if j + 1 < img.shape[1]:
            if class_map[i, j] == class_map[i, j + 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
    else:
        raise ValueError('neighbourhood parameter is wrong!')

    if neighbourhood == 'eight':
        if i - 1 >= 0 and j - 1 >= 0:
            if class_map[i, j] == class_map[i - 1, j - 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i - 1 >= 0 and j + 1 < img.shape[1]:
            if class_map[i, j] == class_map[i - 1, j + 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i + 1 < img.shape[0] and j - 1 >= 0:
            if class_map[i, j] == class_map[i + 1, j - 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i + 1 < img.shape[0] and j + 1 < img.shape[1]:
            if class_map[i, j] == class_map[i + 1, j + 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta

    # add doubleton potential for pixel index_ij and its neighbours with new label
    if neighbourhood == 'four' or neighbourhood == 'eight':
        if i - 1 >= 0:
            if new_label == class_map[i - 1, j]:
                u_potential += -beta
            else:
                u_potential += beta
        if i + 1 < img.shape[0]:
            if new_label == class_map[i + 1, j]:
                u_potential += -beta
            else:
                u_potential += beta
        if j - 1 >= 0:
            if new_label == class_map[i, j - 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if j + 1 < img.shape[1]:
            if new_label == class_map[i, j + 1]:
                u_potential += -beta
            else:
                u_potential += beta
    else:
        raise ValueError('neighbourhood parameter is wrong!')

    if neighbourhood == 'eight':
        if i - 1 >= 0 and j - 1 >= 0:
            if new_label == class_map[i - 1, j - 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if i - 1 >= 0 and j + 1 < img.shape[1]:
            if new_label == class_map[i - 1, j + 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if i + 1 < img.shape[0] and j - 1 >= 0:
            if new_label == class_map[i + 1, j - 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if i + 1 < img.shape[0] and j + 1 < img.shape[1]:
            if new_label == class_map[i + 1, j + 1]:
                u_potential += -beta
            else:
                u_potential += beta

    # return potential
    return u_potential


def calculate_potential_multi_features(img_multi_features, gaussians, class_map, neighbourhood, beta):
    """
    This function gets an image(multiple features), p(features|class i) and predicted label
    of each pixel, then it calculates potential of labeled image
    :param img_multi_features: input image
    :param gaussians: p(features|class i), which is a gaussian distribution
    :param class_map: predicted label for each pixel
    :param neighbourhood: if it be 'four', it will consider 4 of its neighbours,
                        if it be 'eight', it will consider 8 of neighbours.
    :param beta: weight of neighbours
    :return: return a scalar which is potential of labeled image
    """
    import numpy as np

    # potential of labeled image
    u_potential = np.float64(0)
    # potential for singleton
    u_singleton = np.float64(0)
    # potential for doubleton
    u_doubleton = np.float64(0)

    # iterate all pixels
    for i in range(0, img_multi_features.shape[0]):
        for j in range(0, img_multi_features.shape[1]):

            # calculate singleton potential for pixel i,j
            temp = img_multi_features[i, j, :]-gaussians[class_map[i, j]]['mean']
            temp = temp.reshape([1, temp.shape[0]])
            singleton = np.log((np.sqrt((2*np.pi)**img_multi_features.shape[2]*np.linalg.det(gaussians[class_map[i, j]]['std'])))) +\
                           (0.5*np.dot(np.dot(temp, np.linalg.pinv(gaussians[class_map[i, j]]['std'])), np.transpose(temp)))
            u_singleton += singleton[0, 0]
            # print(u_singleton)

            # calculate doubleton potential for pixel i,j and its neighbours
            if neighbourhood == 'four' or neighbourhood == 'eight':
                if i-1 >= 0:
                    if class_map[i,j] == class_map[i-1, j]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i+1 < img_multi_features.shape[0]:
                    if class_map[i, j] == class_map[i+1, j]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if j-1 >= 0:
                    if class_map[i,j] == class_map[i, j-1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if j+1 < img_multi_features.shape[1]:
                    if class_map[i, j] == class_map[i, j+1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
            else:
                raise ValueError('neighbourhood parameter is wrong!')

            if neighbourhood == 'eight':
                if i-1 >= 0 and j-1 >= 0:
                    if class_map[i,j] == class_map[i-1, j-1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i-1 >= 0 and j + 1 < img_multi_features.shape[1]:
                    if class_map[i, j] == class_map[i-1, j+1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i+1 < img_multi_features.shape[0] and j-1 >= 0:
                    if class_map[i,j] == class_map[i+1,j-1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta
                if i+1 < img_multi_features.shape[0] and j+1 < img_multi_features.shape[1]:
                    if class_map[i, j] == class_map[i+1, j+1]:
                        u_doubleton += -beta
                    else:
                        u_doubleton += beta

    # potential of input labeled image
    u_potential = u_singleton + u_doubleton

    # return potential
    return u_potential


def calculate_potential_by_modification_multi_features(img_multi_features, gaussians, class_map, neighbourhood, beta, index_ij, new_label, potential):
    """
    This function doesn't calculate potential from scratch, it calculates
    potential of new stated which obtained by modifying label of one pixel in effective
    way. it calculates difference of changing label of specific pixel and updates
    potential based on potential before changing label of pixel.
    :param img: input image, range of grayscale is in range [0, 1]
    :param gaussians: p(feature|class i), which is a gaussian distribution
    :param class_map: predicted label for each pixel
    :param neighbourhood: if it be 'four', it will consider 4 of its neighbours,
                        if it be 'eight', it will consider 8 of neighbours.
    :param beta: weight of neighbours
    :param index_ij: pixel which its label is proposed to change
    :param new_label: new label of pixel_ij
    :param potential: potential of labeled image before changing the label of pixel_ij
    :return: return a scalar which is potential of new labeled image
    """
    import numpy as np

    u_potential = potential

    i = index_ij[0]
    j = index_ij[1]

    # subtract singleton potential of pixel index_ij
    temp = img_multi_features[i, j, :] - gaussians[class_map[i, j]]['mean']
    temp = temp.reshape([1, temp.shape[0]])
    singleton = np.log((np.sqrt((2*np.pi)**img_multi_features.shape[2]*np.linalg.det(gaussians[class_map[i, j]]['std'])))) +\
                           (0.5*np.dot(np.dot(temp, np.linalg.pinv(gaussians[class_map[i, j]]['std'])), np.transpose(temp)))
    u_potential -= singleton[0, 0]

    # add singleton potential of pixel index_ij with new label
    temp = img_multi_features[i, j, :] - gaussians[new_label]['mean']
    temp = temp.reshape([1, temp.shape[0]])
    singleton = np.log(
        np.sqrt(((2 * np.pi) ** img_multi_features.shape[2] * np.linalg.det(gaussians[new_label]['std'])))) + \
                   (0.5 * np.dot(np.dot(temp, np.linalg.pinv(gaussians[new_label]['std'])), np.transpose(temp)))
    u_potential += singleton[0, 0]

    i = index_ij[0]
    j = index_ij[1]
    # subtract doubleton potential for pixel index_ij and its neighbours
    if neighbourhood == 'four' or neighbourhood == 'eight':
        if i - 1 >= 0:
            if class_map[i, j] == class_map[i - 1, j]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i + 1 < img_multi_features.shape[0]:
            if class_map[i, j] == class_map[i + 1, j]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if j - 1 >= 0:
            if class_map[i, j] == class_map[i, j - 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if j + 1 < img_multi_features.shape[1]:
            if class_map[i, j] == class_map[i, j + 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
    else:
        raise ValueError('neighbourhood parameter is wrong!')

    if neighbourhood == 'eight':
        if i - 1 >= 0 and j - 1 >= 0:
            if class_map[i, j] == class_map[i - 1, j - 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i - 1 >= 0 and j + 1 < img_multi_features.shape[1]:
            if class_map[i, j] == class_map[i - 1, j + 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i + 1 < img_multi_features.shape[0] and j - 1 >= 0:
            if class_map[i, j] == class_map[i + 1, j - 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta
        if i + 1 < img_multi_features.shape[0] and j + 1 < img_multi_features.shape[1]:
            if class_map[i, j] == class_map[i + 1, j + 1]:
                 u_potential -= -beta
            else:
                 u_potential -= beta

    # add doubleton potential for pixel index_ij and its neighbours with new label
    if neighbourhood == 'four' or neighbourhood == 'eight':
        if i - 1 >= 0:
            if new_label == class_map[i - 1, j]:
                u_potential += -beta
            else:
                u_potential += beta
        if i + 1 < img_multi_features.shape[0]:
            if new_label == class_map[i + 1, j]:
                u_potential += -beta
            else:
                u_potential += beta
        if j - 1 >= 0:
            if new_label == class_map[i, j - 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if j + 1 < img_multi_features.shape[1]:
            if new_label == class_map[i, j + 1]:
                u_potential += -beta
            else:
                u_potential += beta
    else:
        raise ValueError('neighbourhood parameter is wrong!')

    if neighbourhood == 'eight':
        if i - 1 >= 0 and j - 1 >= 0:
            if new_label == class_map[i - 1, j - 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if i - 1 >= 0 and j + 1 < img_multi_features.shape[1]:
            if new_label == class_map[i - 1, j + 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if i + 1 < img_multi_features.shape[0] and j - 1 >= 0:
            if new_label == class_map[i + 1, j - 1]:
                u_potential += -beta
            else:
                u_potential += beta
        if i + 1 < img_multi_features.shape[0] and j + 1 < img_multi_features.shape[1]:
            if new_label == class_map[i + 1, j + 1]:
                u_potential += -beta
            else:
                u_potential += beta

    # return potential
    return u_potential