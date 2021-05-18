from utils.resize_multichannel_image import resize_multichannel_image

def augment_resize(sample_data, target_size, order=3):
    """
    Reshapes data (and seg) to target_size
    :param sample_data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param target_size: int or list/tuple of int
    :param order: interpolation order for data (see skimage.transform.resize)
    :return:
    """
    dimensionality = len(sample_data.shape) - 1
    if not isinstance(target_size, (list, tuple)):
        target_size_here = [target_size] * dimensionality
    else:
        assert len(target_size) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                   "the same dimensionality as data!"
        target_size_here = list(target_size)

    sample_data = resize_multichannel_image(sample_data, target_size_here, order)

    return sample_data