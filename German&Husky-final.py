#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[3]:


import keras
print(tf.__version__)


# In[4]:


import os
from os import mkdir
import shutil
from shutil import copyfile 
import image
import PIL


# In[5]:






# The path to the directory where the original
# dataset was uncompressed


#original_dataset_dir = '/content/drive/My Drive/train/train'

original_dataset_dir='D:\\0-Fall-2020\\Machine learning\\Imagenet datasets\\main-dataset'

# The directory where we will
# store our smaller dataset


#base_dir = '/content/drive/My Drive/new_sets'

base_dir = 'D:\\0-Fall-2020\\Machine learning\\Imagenet datasets\\new_sets'

#os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

# Directory with our training german pictures
train_german_dir = os.path.join(train_dir, 'german')
#os.mkdir(train_german_dir)

# Directory with our training husky pictures
train_husky_dir = os.path.join(train_dir, 'husky')
#os.mkdir(train_husky_dir)

# Directory with our validation german pictures
validation_german_dir = os.path.join(validation_dir, 'german')
#os.mkdir(validation_german_dir)

# Directory with our validation husky pictures
validation_husky_dir = os.path.join(validation_dir, 'husky')
#os.mkdir(validation_husky_dir)

# Directory with our validation german pictures
test_german_dir = os.path.join(test_dir, 'german')
#os.mkdir(test_german_dir)

# Directory with our validation husky pictures
test_husky_dir = os.path.join(test_dir, 'husky')
#os.mkdir(test_husky_dir)
'''
# Copy first 1200 german images to train_german_dir

fnames = ['german ({}).jpeg'.format(i) for i in range(1,1200)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_german_dir, fname)
    copyfile(src, dst)

# Copy next 270 german images to validation_german_dir
fnames = ['german ({}).jpeg'.format(i) for i in range(1200, 1470)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_german_dir, fname)
    copyfile(src, dst)

# Copy next 270 german images to test_german_dir
fnames = ['german ({}).jpeg'.format(i) for i in range(1470, 1741)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_german_dir, fname)
    copyfile(src, dst)

# Copy first 1100 husky images to train_husky_dir
fnames = ['husky ({}).jpeg'.format(i) for i in range(1,1100)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_husky_dir, fname)
    copyfile(src, dst)

# Copy next 221 husky images to validation_husky_dir
fnames = ['husky ({}).jpeg'.format(i) for i in range(1100, 1321)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_husky_dir, fname)
    copyfile(src, dst)

# Copy next 221 husky images to test_husky_dir
fnames = ['husky ({}).jpeg'.format(i) for i in range(1321, 1542)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_husky_dir, fname)
    copyfile(src, dst)

'''    


# In[6]:


print('number german images for  training :', len(os.listdir(train_german_dir)))
print('number husky images for training :', len(os.listdir(train_husky_dir)))
print('number german images for validation :', len(os.listdir(validation_german_dir)))
print('number husky images for validation :', len(os.listdir(validation_husky_dir)))
print('number german images for test :', len(os.listdir(test_german_dir)))
print('number husky  for images test :', len(os.listdir(test_husky_dir)))


# In[21]:


#Standardize the data
from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train = ImageDataGenerator(rescale=1./255)
val = ImageDataGenerator(rescale=1./255)

train_generator = train.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = val.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')


# In[22]:


#build model
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()


# In[23]:


from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[35]:


get_ipython().system('pip install pillow')


# In[15]:


get_ipython().system('pip install matplotlib')


# In[24]:


#fit model 
import PIL
#history = model.fit_generator(
history = model.fit(
      train_generator,
      steps_per_epoch=50,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=30)


# In[25]:


import matplotlib.pyplot as plt
def plot_graphs(fname, history, metric):
    plt.plot(history.history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()
    plt.close()

plot_graphs("loss", history, 'loss')
plot_graphs("acc", history, 'acc')


# In[26]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[27]:


test_generator = val.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=12,
        class_mode='binary')

model.evaluate(test_generator) 

