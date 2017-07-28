import os

import numpy as np
import scipy.ndimage
from skimage.transform import resize

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras import backend

from .model import RadioModel

from PIL import Image

MODEL_DIR = 'saved_models'


class ModelHearthSegmentation(RadioModel):
    def __init__(self):
        super(ModelHearthSegmentation, self).__init__()

    def predict(self, image):
        model2 = self.get_unet()
        path = os.path.join(os.path.dirname(__file__), MODEL_DIR, 'keras_model_heart_weights')
        model2.load_weights(path)


        images_test = scipy.misc.imread(image)
        images_test = scipy.misc.imresize(images_test[:, :, 1], [96, 96])
        images_test = images_test[..., np.newaxis]
        images_test = images_test[np.newaxis, ...]
        images_test = images_test.astype('float32')

        mean = np.mean(images_test)     # mean for data centering
        std = np.std(images_test)       # std for data normalization

        images_test -= mean
        images_test /= std

        predicted = model2.predict(images_test)
        images_test = images_test[0, :, :, 0] # back to normal 96 x 96 array
        predicted = predicted[0,:,:,0]

        images_test *= std              # returns back from normalization
        images_test += mean             # returns back from normalization

        uploaded_image = np.zeros(shape=(96, 96, 3), dtype=float) # defines new array with 3 channels
        uploaded_image[:, :, 0] = images_test
        uploaded_image[:, :, 1] = images_test
        uploaded_image[:, :, 2] = images_test

        predicted2 = np.zeros(shape=(96, 96, 4), dtype=float)   # 4 channels, since transparency needed on mask
        predicted2[:, :, 0] = predicted * 255                   # 255 max value in 8 bit images
        predicted2[:, :, 3] = predicted * 255 / 5               # Alpha/transparency channel

        uploaded_image = np.asarray(images_test, np.uint8)      # this fucker took long to find
        predicted2 = np.asarray(predicted2, np.uint8)           # changes type to that "Image" will take it

        background = Image.fromarray(uploaded_image)
        foreground = Image.fromarray(predicted2)

        foreground = foreground.convert("RGBA")                 # this too needs to be done for some reason
        background = background.convert("RGBA")

        background = Image.alpha_composite(background, foreground) # the essence/magic, merges the images

        merged_image = np.asarray(background)                   # back to array


        return (uploaded_image, merged_image)

        # plt.imshow(images_test[0, :, :, 0], cmap='Greys_r')
        # plt.axis('off')
        # res = pred[0, :, :, 0]
        # res[res <= 0.9] = np.nan
        # plt.savefig('raw.png')
        # plt.imshow(pred[0, :, :, 0], alpha=0.5, cmap='Reds')
        # plt.savefig('heart.png')
        #
        # # return plt.show()

    def get_unet(self, lr=1e-4, img_rows=96, img_cols=96):
        inputs = Input((img_rows, img_cols, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        model.compile(optimizer=Adam(lr=lr), loss=self.dice_coef_loss, metrics=[self.dice_coef])

        return model

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1.):
        y_true_f = backend.flatten(y_true)
        y_pred_f = backend.flatten(y_pred)
        intersection = backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

    @staticmethod
    def pre_process(imgs, img_rows=96, img_cols=96):
        """Unused"""
        images_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
        for i in range(imgs.shape[0]):
            images_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
        images_p = images_p[..., np.newaxis]
        return images_p
