import keras
import numpy as np
import scipy.misc
from keras.models import load_model

def verycool():
    
    model = load_model('model/kerasmodel')

    return (model.predict_classes(np.reshape(scipy.misc.imread("model/outfile.jpg"),[1,100,100,3]), verbose=0))