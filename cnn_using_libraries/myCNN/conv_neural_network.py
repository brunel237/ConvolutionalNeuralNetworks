import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import warnings
from PIL import Image 
warnings.filterwarnings('ignore')
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import image_dataset_from_directory
#from keras.preprocessing.image import ImageDataGenerator, load_img

import os

#PATH = '/media/brunel/ubuntu_xtend2/Devoir242/myCNN/img_data'
save_path = '/media/brunel/ubuntu_xtend2/Devoir242/historique'
path = '/media/brunel/ubuntu_xtend2/Devoir242/training_set'
classes = os.listdir(path)
print(classes)

base_dir = '../../DataSet/training_set'
sec_dir = '../../DataSet/test_set'

train = image_dataset_from_directory(base_dir, image_size=(64,64), subset='training', validation_split=0.1,seed = 1,  batch_size= 32)
test = image_dataset_from_directory(sec_dir, image_size=(64,64), subset='validation', validation_split=0.1,seed = 1, batch_size= 32)

print(train)

model = tf.keras.models.load_model(save_path)
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu',),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

cat_dir = os.path.join('img_data/cat')
dog_dir = os.path.join('img_data/dog')
cat_names = os.listdir(path+'/cats')
dog_names = os.listdir(path+'/dogs')
i=6335
for x in dog_names:
    t = Image.open(path+'/dogs/'+x)
    t = t.convert('L')
    t = t.rotate(200)
    t=t.resize((500, 500))
    t.save("new_dog"+str(i)+".png", 'png')
    i+=1
    if i == 8334:
         break
i=0
for x in dog_names:
    t = Image.open(path+'/dogs/'+x)
    t = t.rotate(90)
    t.save("new_dog"+str(i), 'png')
    i+=1


check_point = tf.keras.callbacks.ModelCheckpoint(filepath='.', save_weights_only=True, verbose=1)

history = model.fit(
    train, 
    epochs=25, 
    validation_data=test, 
    callbacks=[check_point]
)

test_loss, accuracy = model.evaluate(test, verbose=1)
print('LOSS : ',test_loss)
print('ACC : ',accuracy)

#l'historique
model.save(save_path)


def prediction(path):
    test_image = Image.open(path)
    test_image = test_image.resize((64, 64))
    test_image = np.asarray(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)

    if(result >= 0.5):
        print("Dog -- ",result)
    else:
	    print("Cat -- ",result)
