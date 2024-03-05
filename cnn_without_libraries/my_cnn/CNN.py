import fonctions as f
from PIL import Image
import os
import numpy as np
import csv
from tqdm import tqdm

train_path = os.path.join('/media/brunel/ubuntu_xtend2/Devoir242/training_set/')

cat_path = os.listdir(train_path+'cats')
dog_path = os.listdir(train_path+'dogs')


poids = []
biais = []
filtres = []

itr = 0
for image in tqdm(dog_path):
    img = Image.open(train_path+'dogs/'+image)
    img = img.resize((64, 64))
    img = img.convert("L")
    img = np.asarray(img)
    t = img.copy()
    for i in range(64):
        for j in range(64):
            t[i][j] /= (10000)

    #print(img.shape)
    itr += 1
    poids, biais, filtres = f.entrainement([t], 1, poids, biais, filtres, 5)
    if itr == 11:
        break

itr = 0
for image in tqdm(cat_path):
    img = Image.open(train_path+'cats/'+image)
    img = img.resize((64, 64))
    img = img.convert("L")
    img = np.asarray(img)
    t = img.copy()
    for i in range(64):
        for j in range(64):
            t[i][j] /= (255) * 10
    #print(img.shape)
    itr += 1
    poids, biais, filtres = f.entrainement([t], 0, poids, biais, filtres, 5)
    if itr == 11:
        break

path = '/media/brunel/ubuntu_xtend2/Devoir242/training_set/cats/cat.1600.jpg'

print ( f.prediction(path, poids, biais, filtres))







