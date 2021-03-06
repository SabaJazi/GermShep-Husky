# Reference Deep learning with Python (Francois Chollet)
import os
from os import mkdir
import shutil
from shutil import copyfile
from os import path
#import keras
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# # section1 :  partition data

#directory of working data
work_dir = '/Users/TAMDO/PythonProject/ML5215/TamDogvscat/train/train'


# store our  new partition  dataset in this directory
partition_dir = '/Users/TAMDO/PythonProject/ML5215/TamDogvscat/TamDC'
if not path.isdir(partition_dir):
  os.mkdir(partition_dir)

# Directories for our train,validation and test
train_dir = path.join(partition_dir, 'train')
if not path.isdir(train_dir):
  os.mkdir(train_dir)
validation_dir = os.path.join(partition_dir, 'validation')
if not path.isdir(validation_dir):
  os.mkdir(validation_dir)
test_dir = path.join(partition_dir, 'test')
if not path.isdir(test_dir):
  os.mkdir(test_dir)

# train -> cats
train_cats_dir = path.join(train_dir, 'cats')
if not path.isdir(train_cats_dir):
  os.mkdir(train_cats_dir)

# train -> dogs
train_dogs_dir = path.join(train_dir, 'dogs')
if not path.isdir(train_dogs_dir):
  os.mkdir(train_dogs_dir)

# validation ->cats
validation_cats_dir = path.join(validation_dir, 'cats')
if not path.isdir(validation_cats_dir):
  os.mkdir(validation_cats_dir)

# validation ->dogs
validation_dogs_dir = path.join(validation_dir, 'dogs')
if not path.isdir(validation_dogs_dir):
  os.mkdir(validation_dogs_dir)

#  test ->cats
test_cats_dir = path.join(test_dir, 'cats')
if not path.isdir(test_cats_dir):
  os.mkdir(test_cats_dir)

#  test ->dogs
test_dogs_dir = path.join(test_dir, 'dogs')
if not path.isdir(test_dogs_dir):
  os.mkdir(test_dogs_dir)

# Copy first 10000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(10000)]
for fname in fnames:
    src = path.join(work_dir, fname)
    dst = path.join(train_cats_dir, fname)
    if not path.exists(dst):copyfile(src, dst)

# Copy next 1250 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(10000, 11250)]
for fname in fnames:
    src = path.join(work_dir, fname)
    dst = path.join(validation_cats_dir, fname)
    if not path.exists(dst):copyfile(src, dst)

# Copy next 1250 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(11250,12500)]
for fname in fnames:
    src = os.path.join(work_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    if not path.exists(dst):copyfile(src, dst)

# Copy first 10,000 dog images from original trainset to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(10000)]
for fname in fnames:
    src = path.join(work_dir, fname)
    dst = path.join(train_dogs_dir, fname)
    if not path.exists(dst):copyfile(src, dst)

# Copy next 1,250 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(10000, 11250)]
for fname in fnames:
    src = path.join(work_dir, fname)
    dst = path.join(validation_dogs_dir, fname)
    if not path.exists(dst):copyfile(src, dst)

# Copy next ,1250 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(11250, 12500)]
for fname in fnames:
    src = os.path.join(work_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    if not path.exists(dst):copyfile(src, dst)

# Section2: Standardize and augmentation data


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
        batch_size=20,
        class_mode='binary')
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='binary')

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=64,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary')




# Section3: build model 4
# model 1 drop 1 conv2D, drop 1 Maxpooling2D, add 1 Dropout



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Section 4:fit model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=500,
      validation_data=validation_generator,
      validation_steps=50)


#Section5: draw graph to demontrate loss, accuray of this model
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
v_loss = history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.xlabel('epochs')
#plt.plot(range(epoch_num, next_epoch), g["ta"], 'g', label='Training acc model 1',)
plt.plot(epochs, acc, 'g', label='Training acc model 4',)
#plt.plot(range(epoch_num, next_epoch), g["va"], 'b', label='Validation acc model 1')
plt.plot(epochs, val_acc, 'b', label='Validation acc model 4')
#epoch_num = next_epoch
plt.title('Training and validation Accuracy model 4')
plt.legend()
plt.figure()
plt.show()
plt.grid(True)

plt.xlabel('epoch')

#plt.plot(range(epoch_num, next_epoch), g["tl"], '--', label='Training loss model 1')
plt.plot(epochs, loss, '--', label='Training loss model 4')
plt.plot(epochs, v_loss, 'b', label='Validation loss model 4')
#epoch_num = next_epoch
plt.title('Training and validation Loss model 4')
plt.legend()
plt.show()
plt.grid(True)


# section 6:  evaluate model 4, using test data
test_loss,test_acc=model.evaluate_generator(test_generator,steps=50)
print ('\n The  accuracy of model 4 using test data : ', test_acc)


