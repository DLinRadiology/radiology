import keras
import numpy as np
import scipy.misc
from keras.models import load_model
from keras import backend as K

def verycool(image):
    
    
    image = scipy.misc.imread(image)
    image = scipy.misc.imresize(image, [100,100,3])

    arr_min = np.min(image)
    arr_max = np.max(image)
    normalized = (image - arr_min) / (arr_max - arr_min + K.epsilon())
    image = normalized
    
    
    
    model = load_model('model/kerasmodel')

#a = (model.predict_classes(np.reshape(scipy.misc.imread("model/outfile.jpg"),[1,100,100,3]), verbose=0))
    a = (model.predict_classes(np.reshape(image,[1,100,100,3]), verbose=0))
    if (a == 1):
        return ("it´s a frontal image")
    if (a == 0):
        return ("it´s a lateral image")


"""def normalize(array, min_value=0., max_value=1.):
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + K.epsilon())
    return (max_value - min_value) * normalized + min_value"""

#print(verycool())
