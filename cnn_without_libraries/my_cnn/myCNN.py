import imageio.v3 as iio
from convolutional import *
from activations import *
from import_class import *
from dense import *
from reshape import *
from mnist import *
from pathlib import Path

train_path = "/media/brunel/ubuntu_xtend2/Devoir242/training_set"
test_path = "/media/brunel/ubuntu_xtend2/Devoir242/test_set"

trainI = GetTrainImage(PATH=train_path, IMAGE_SIZE=64)
testI = GetTrainImage(PATH=test_path, IMAGE_SIZE=64)

images = list()

i=1

for file in Path(train_path+'/cats/').iterdir():
    images.append(iio.imread(file))
    if i==10:
        break

print(images)

(train_images, train_labels) = trainI.load_dataset()
(test_images, test_labels) = testI.load_dataset()

print('train = ',train_labels)
print('test = ',test_labels)

(train_images, train_labels) = preprocess_data(train_images, train_labels, 15000)
(test_images, test_labels) = preprocess_data(test_images, test_labels, 5000)



print('train = ',train_labels)
print('test = ',test_labels)

