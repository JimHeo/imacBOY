import numpy as np
from skimage.transform import resize

def resize_multichannel_image(multichannel_image, new_shape, order=3):
    '''
    Resizes multichannel_image. Resizes each channel in c separately and fuses results back together

    :param multichannel_image: c x x x y (x z)
    :param new_shape: x x y (x z)
    :param order:
    :return:
    '''
    tpe = multichannel_image.dtype
    new_shp = [multichannel_image.shape[0]] + list(new_shape)
    result = np.zeros(new_shp, dtype=multichannel_image.dtype)
    for i in range(multichannel_image.shape[0]):
        result[i] = resize(multichannel_image[i].astype(float), new_shape, order, "constant", 0, True, anti_aliasing=False)
    return result.astype(tpe)