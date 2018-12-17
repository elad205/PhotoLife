from keras.layers import Conv2D
from scipy.misc import imsave
import keras
from skimage import color
import numpy as np
NUM_OUTPUT_FILTERS = 25
KERNEL_SIZE = (3, 3)
image_size = 200
""""
img = skimage.io.imread("a.png")
img = transform.resize(img, (400, 400))
lab = color.rgb2lab(1.0/255*img)
NUM_OF_SAMPLES = 1
features = lab[:, :, 0]
labels = lab[:, :, 1:]
features = features.reshape(NUM_OF_SAMPLES, 400, 400, 1)
labels = labels.reshape(NUM_OF_SAMPLES, 400, 400, 2)
"""
image = keras.preprocessing.image.img_to_array(keras.preprocessing.image.
                                               load_img('a.png'))
image = np.array(image, dtype=float)

# Import map images into the lab colorspace
features_x = color.rgb2lab(1.0 / 255 * image)[:, :, 0]
labels_y = color.rgb2lab(1.0/255*image)[:, :, 1:]
labels_y = labels_y / 128
features_x = features_x.reshape(1, 400, 400, 1)
labels_y = labels_y.reshape(1, 400, 400, 2)


model = keras.models.Sequential()
model.add(keras.engine.InputLayer((None, None, 1)))


def conv_layer(fl):
    model.add(Conv2D(fl, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))


def deconv_layer(fl):
    model.add(Conv2D(fl, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.UpSampling2D())


num_of_filters = 8
num_of_conv_blocks = 4
num_of_deconv_blocks = num_of_conv_blocks - 1
for _ in range(num_of_conv_blocks):
    conv_layer(num_of_filters)
    num_of_filters = num_of_filters * 2

model.add(keras.layers.UpSampling2D())
num_of_filters = 32
for _ in range(num_of_deconv_blocks):
    deconv_layer(num_of_filters)
    num_of_filters = int(num_of_filters / 2)

model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

model.compile(optimizer="rmsprop", loss='mse')
for layer in model.layers:
    print(layer.output_shape)

model.fit(x=features_x, y=labels_y, epochs=3000, batch_size=1)
scores = model.evaluate(features_x, labels_y)
print(scores * 100)

output = model.predict(features_x)
output = output * 128
canvas = np.zeros((400, 400, 3))
canvas[:, :, 0] = features_x[0][:, :, 0]
canvas[:, :, 1:] = output[0]
imsave("img_result.png", color.lab2rgb(canvas))
imsave("img_gray_scale.png", color.rgb2gray(color.lab2rgb(canvas)))
