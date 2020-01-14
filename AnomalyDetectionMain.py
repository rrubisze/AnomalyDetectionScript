#!/usr/bin/env python
# coding: utf-8

# # GOOGLECOLAB NOTEBOOK

# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from keras.utils import np_utils

from PIL import Image
import requests
from io import BytesIO
import os
import pickle
import tqdm
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# In[ ]:


### SET COLAB ENVIROMENT ###

# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)


# In[ ]:


### READ DATA ###

df = pd.read_json('Crack.json', lines=True)
df['label'] = df.annotation.apply(lambda x: x['labels'][0] if len(x['labels']) == 1 else 'Crack')
print(df.shape)
df.head()

# In[ ]:


### GET AND PREPARE IMAGES ###

images = []

for url in tqdm.tqdm(df['content']):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    numpy_img = img_to_array(img)
    img_batch = np.expand_dims(numpy_img, axis=0)
    images.append(img_batch.astype('float16'))

images = np.vstack(images)
print(images.shape)

# In[ ]:


# In[ ]:


### RANDOM IMAGES PLOT ###

random_id = np.random.randint(0, images.shape[0], 4)
f, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(16, 10))

for ax, img, title in zip(axes.ravel(), images[random_id], df['label'][random_id]):
    ax.imshow(array_to_img(img))
    ax.set_title(title)

# In[ ]:


### IMPORT VGG16 ###

vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# In[ ]:


# Freeze the layers except the last 2 convolutional blocks
for layer in vgg_conv.layers[:-8]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# In[ ]:


### ENCODE LABEL ###

Y = np_utils.to_categorical((df.label.values == 'Crack') + 0)

# In[ ]:


### CREATE TRAIN TEST ###

X_train, X_test, y_train, y_test = train_test_split(images, Y, random_state=42, test_size=0.2)

# In[ ]:


### MODIFY VGG STRUCTURE ###

x = vgg_conv.output
x = GlobalAveragePooling2D()(x)
x = Dense(2, activation="softmax")(x)

model = Model(vgg_conv.input, x)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model.summary()

# In[ ]:


### INITIALIZE TRAIN GENERATOR ##

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

# In[ ]:


train_datagen.fit(X_train)

# In[ ]:


model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=2)

# In[ ]:


print(classification_report(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test / 255), axis=1)))


# In[ ]:


# In[ ]:


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)


# In[ ]:


cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test / 255), axis=1))
plt.figure(figsize=(7, 7))
plot_confusion_matrix(cnf_matrix, classes=df.label.unique(), title="Confusion matrix")
plt.show()


# In[ ]:


# In[ ]:


# In[ ]:


### CREATE FUNCTION TO DRAW ANOMALIES ###

def plot_activation(img):
    pred = model.predict(img[np.newaxis, :, :, :])
    pred_class = np.argmax(pred)

    weights = model.layers[-1].get_weights()[0]
    class_weights = weights[:, pred_class]

    intermediate = Model(model.input, model.get_layer("block5_conv3").output)
    conv_output = intermediate.predict(img[np.newaxis, :, :, :])
    conv_output = np.squeeze(conv_output)

    h = int(img.shape[0] / conv_output.shape[0])
    w = int(img.shape[1] / conv_output.shape[1])

    activation_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)
    out = np.dot(activation_maps.reshape((img.shape[0] * img.shape[1], 512)), class_weights).reshape(img.shape[0],
                                                                                                     img.shape[1])

    plt.imshow(img.astype('float32').reshape(img.shape[0], img.shape[1], 3))
    plt.imshow(out, cmap='jet', alpha=0.35)
    plt.title('Crack' if pred_class == 1 else 'No Crack')

# In[ ]:


# plot_activation(X_test[200] / 255)
#
# # In[ ]:
#
#
# plot_activation(X_test[100] / 255)
#
# # In[ ]:
#
#
# plot_activation(X_test[134] / 255)

plot_activation(img_to_array(images[0])/255)

plot_activation(img_to_array(images[0])/255)
plot_activation(img_to_array(images[0])/255)
plot_activation(img_to_array(images[0])/255)
# In[ ]:




