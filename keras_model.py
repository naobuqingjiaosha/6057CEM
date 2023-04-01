import keras
import cv2
import numpy as np

import os
import tensorflow as tf
dir = 'flower\Dataset\\'
sunflower_dir = dir + 'sunflower'
tulip_dir = dir + 'tulip'
rose_dir = dir + 'rose'
dandelion_dir = dir + 'dandelion'
daisy_dir = dir + 'daisy'
X = []
y_label = []
imgsize = 150

def training_data(label, dir):
    print('loading:', dir)
    for img in os.listdir(dir):
        path = os.path.join(dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))
        X.append(np.array(img))
        y_label.append(np.array(str(label)))


training_data('sunflower', sunflower_dir)
training_data('tulip', tulip_dir)
training_data('rose', rose_dir)
training_data('dandelion', dandelion_dir)
training_data('daisy', daisy_dir)

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)
y = to_categorical(y,5)
X = np.array(X)
X = X/255

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from keras import layers
from keras import models
from keras import optimizers
cnn = models.Sequential()
cnn.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3))) #convolution
cnn.add(layers.MaxPooling2D((3,3)))#pooling
cnn.add(layers.Conv2D(64,(3,3),activation='relu'))
cnn.add(layers.Dropout(0.5))
cnn.add(layers.MaxPooling2D((3,3)))
cnn.add(layers.Conv2D(128,(3,3),activation='relu'))
cnn.add(layers.Dropout(0.5))
cnn.add(layers.MaxPooling2D((2,2)))
cnn.add(layers.Conv2D(256,(3,3),activation='relu'))
cnn.add(layers.MaxPooling2D((3,3)))
cnn.add(layers.Flatten())
cnn.add(layers.Dropout(0.5))
cnn.add(layers.Dense(512,activation='relu'))
cnn.add(layers.Dense(5,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(learning_rate=0.0001),metrics=['acc'])

# history = cnn.fit(X_train,y_train,epochs=500,batch_size=8,validation_data=(X_test,y_test))

#打印模型架构
print(cnn.summary())

from keras.preprocessing.image import ImageDataGenerator
augs_gen = ImageDataGenerator(featurewise_center=False,
                              samplewise_center=False,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=False,
                              rotation_range=50,
                              zoom_range=0.1,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=False)
#augs_gen.fit(X_train)
history = cnn.fit_generator(augs_gen.flow(X_train,y_train,batch_size=16),
                            validation_data=(X_test,y_test),
                            validation_steps=10,
                            steps_per_epoch=100,
                            epochs=100,
                            verbose=1)
#cnn.save('keras_model.h5')

#plot
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()