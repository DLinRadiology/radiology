from collections import Sequence

import keras
import numpy as np
import scipy.misc
from keras.models import load_model
from keras import backend


def verycool(image):
    image = scipy.misc.imread(image)
    image = scipy.misc.imresize(image, [100, 100, 3])
    image = normalize(image)

    model = load_model('model/kerasmodel')
    a = model.predict_classes(np.reshape(image, [1, 100, 100, 3]), verbose=0)
    return "it´s a {} image".format('frontal' if a == 1 else 'lateral')


def normalize(array, min_max_values=None):
    """Normalize array

    Parameters
    ----------
    array : np.ndarray
    min_max_values : Sequence, optional
        if not None, then will bound values to range specified by tuple or list

    Returns
    -------
    normalized : np.ndarray
    """
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + backend.epsilon())
    if min_max_values is not None:
        assert isinstance(min_max_values, Sequence)
        assert len(min_max_values) == 2
        assert all(isinstance(v, float) for v in min_max_values)
        min_value = min_max_values[0]
        max_value = min_max_values[1]
        normalized = (max_value - min_value) * normalized + min_value
    return normalized

