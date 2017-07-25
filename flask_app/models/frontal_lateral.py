import os

import numpy as np
import scipy.misc
from keras.models import load_model

from .model import RadioModel
from .utils import normalize

MODEL_DIR = 'saved_models'


class ModelFrontalLateral(RadioModel):
    def __init__(self, img_size=(100, 100, 3)):
        super(ModelFrontalLateral, self).__init__()
        self.img_size = img_size

    def predict(self, image):
        """
        Parameters
        ----------
        image : np.ndarray

        Returns
        -------
        prediction : int
            1 - frontal
            0 - lateral
        """
        self.image = image
        image = scipy.misc.imread(image)
        image = scipy.misc.imresize(image, self.img_size)
        image = normalize(image)
        path = os.path.join(os.path.dirname(__file__), MODEL_DIR, 'keras_model_frontal_lateral')
        model = load_model(path)
        shape = (1,) + self.img_size
        return model.predict_classes(np.reshape(image, shape), verbose=0)
