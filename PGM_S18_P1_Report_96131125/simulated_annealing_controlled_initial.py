def simulated_annealing_controlled_initial(img, true_segmented_img, probability_init, gaussians, neighbourhood, beta, initial_temperature, labels_list, max_iter):
    """
    This function uses simulated annealing optimization to minimize potential of
    labeled image.
    :param img: input image, range of gray-scale is in range [0, 1]
    :param true_segmented_img: true labeled image
    :param probability_init: During initialization of labeled image, with probability
                         probability_init, we choose ture label of pixel as its initial label
    :param gaussians:  p(feature|class i), which is a gaussian distribution
    :param neighbourhood:  if it be 'four', it will consider 4 of its neighbours,
                        if it be 'eight', it will consider 8 of neighbours. it's used
                        by potential function.
    :param beta: weight of neighbours. a parameter for potential function
    :param initial_temperature
    :param label_list: a np.array list that contains different labels(classes)
    :param max_iter: maximum number of epochs
    :return: a labeled image.
    """
    import numpy as np
    from calculate_potential import calculate_potential
    from calculate_potential import calculate_potential_by_modification

    np.random.seed(0)

    # initial max iteration
    k_max = max_iter

    # initial temperature
    t0 = initial_temperature
    current_temperature = np.float64(t0)

    # initial labeled image, first initial labels randomly, then
    # for each pixel with probability 'probability_init' select its true label
    # as its initial label and with probability 1-'probability_init' use random label

    # random initialization
    class_map = np.random.randint(labels_list.shape[0], size=img.shape)
    class_map = labels_list[class_map]
    # with the given probability select random label for each
    # pixel or replace the random label with true label
    for i in range(0, class_map.shape[0]):
        for j in range(0, class_map.shape[1]):
            # generate a random number from uniform distribution [0,1)
            rand_num = np.random.uniform(low=0, high=1)

            if rand_num < probability_init:
                class_map[i, j] = true_segmented_img[i, j]


    # current potential
    current_potential = calculate_potential(img=img, gaussians=gaussians, class_map=class_map,
                                            neighbourhood=neighbourhood, beta=beta)

    # minimum potential
    min_potential = current_potential
    min_potential_labeled_map = np.copy(class_map)

    # do k_max iteration for minimizing potential
    for k in range(0, k_max):

        #print('iteration {}/{}...'.format(k, k_max))

        # create a new proposed labeled map which different just in one pixel
        class_map_proposed = np.copy(class_map)
        # randomly choose a pixel
        rand_pixel_i = np.random.randint(class_map.shape[0])
        rand_pixel_j = np.random.randint(class_map.shape[1])
        # choose randomly new label
        new_label = class_map[rand_pixel_i, rand_pixel_j]
        while new_label == class_map[rand_pixel_i, rand_pixel_j]:
            new_label = np.random.randint(labels_list.shape[0])
            new_label = labels_list[new_label]
        class_map_proposed[rand_pixel_i, rand_pixel_j] = new_label

        # potential of new proposed state
        #new_proposed_potential = calculate_potential(img=img, gaussians=gaussians, class_map=class_map_proposed,
        #                                            neighbourhood=neighbourhood, beta=beta)
        new_proposed_potential = calculate_potential_by_modification(img=img, gaussians=gaussians, class_map=class_map
                                                                     , neighbourhood=neighbourhood, beta=beta
                                                                     , index_ij=[rand_pixel_i, rand_pixel_j],
                                                                     new_label=new_label,
                                                                     potential=current_potential)

        # calculate delta U
        delta_u = new_proposed_potential - current_potential

        if delta_u <= 0:
            # if delta u is less equal than 0, accept proposed state
            current_potential = new_proposed_potential
            class_map[rand_pixel_i, rand_pixel_j] = new_label

            # if new proposed labeled map is best keep it
            if min_potential > new_proposed_potential:
                min_potential = new_proposed_potential
                min_potential_labeled_map = np.copy(class_map_proposed)

        else:
            # generate a random number from uniform distribution [0,1)
            rand_num = np.random.uniform(low=0, high=1)

            if rand_num < np.exp(-delta_u/current_temperature):
                # accept new proposed state with a probability
                current_potential = new_proposed_potential
                class_map[rand_pixel_i, rand_pixel_j] = new_label
            else:
                pass

        # update tempreture
        if current_temperature > 0.001:
            current_temperature = 0.95 * current_temperature

    # return best labeled map
    return min_potential_labeled_map