import math as Math
from random import *

def convolution(tableau_image, tableau_filtres):
    nb_filtres = len(tableau_filtres)
    nb_images = len(tableau_image)
    taille_image = len(tableau_image[0])
    taille_filtre = len(tableau_filtres[0])
    profondeur = nb_filtres*nb_images
    taille_sortie = taille_image-taille_filtre +1
    sortie = []

    for i in range(nb_images):
        for j in range(nb_filtres):
            resultat = [[0]*taille_sortie for i in range(taille_sortie)]
            for x in range(taille_sortie):
                for y in range(taille_sortie):
                    som = 0
                    for m in range(taille_filtre):
                        for n in range(taille_filtre):
                            som += tableau_filtres[j][m][n]*tableau_image[i][m+x][n+y]
                    
                    resultat[x][y] = som

            sortie.append(resultat)
    return sortie

def maxPooling(tableau_image):
    nb_images = len(tableau_image)
    taille_image = len(tableau_image[0])

    tab_image_sortie = []

    for k in range(nb_images):
        image_sortie = [[0]*(taille_image//2) for i in range((taille_image//2))]
        for i in range(0, taille_image//2, 2):
            for j in range(0, taille_image//2, 2):
                image_sortie[i][j] = max(max(tableau_image[k][i][j:j+2], tableau_image[k][i+1][j:j+2]))
        tab_image_sortie.append(image_sortie)
    return tab_image_sortie

def reversePooling(tableau_image):
    nb_images = len(tableau_image)
    taille_image = len(tableau_image[0])
    tab_image_sortie = []
    for k in range(nb_images):
        resultat = [[0]*len(tableau_image[0][0])*2 for i in range(taille_image*2)]
        for i in range(taille_image):
            for j in range(len(tableau_image[0][0])):
                resultat[i*2][j*2] = tableau_image[k][i][j]
        tab_image_sortie.append(resultat)
    return tab_image_sortie

def activation_relu(tableau_image):
    for k in range(len(tableau_image)):
        for i in range(len(tableau_image[0])):
            for j in range(len(tableau_image[0])):
                if tableau_image[k][i][j] < 0:
                    tableau_image[k][i][j] = 0.001
    return tableau_image

def reverse_relu(tableau_image):
    for k in range(len(tableau_image)):
        for i in range(len(tableau_image[0])):
            for j in range(len(tableau_image[0])):
                if tableau_image[k][i][j] > 0 and tableau_image[k][i][j] < 0.1:
                    tableau_image[k][i][j] = -1
    return tableau_image

def applatissement(tableau_image):
    sortie = []
    for k in range(len(tableau_image)):
        for i in range(len(tableau_image[0])):
            for j in range(len(tableau_image[0])):
                sortie.append(tableau_image[k][i][j])
    return sortie

def reverse_vectorisation(vecteur):
    matrice = []
    temp=[] ; compt = 0
    for i in range(len(vecteur)):
        compt += 1
        temp.append(vecteur[i])
        matrice.append(temp)
        temp=[]

    return [matrice]

def couche_convolutive_reverse(F, images, classe, image_applatie=[], nb_couche=3):
    filtres =[]
    if image_applatie!=[]:
        A = activation_sigmoide(image_applatie)
        for i in range(len(A)):
            A[i] -= classe
        matrice = reverse_vectorisation(A)
    else:
        matrice = images

    for i in range(nb_couche-1, 0, -1):
        tableau_image_pooling = reversePooling(matrice)
        filtre_couche_i = convolution([images], tableau_image_pooling)
        filtres.append(filtre_couche_i)
        matrice = [images]

    for k in range(len(filtres)):
        for i in range(len(filtres[0])):
            for j in range(len(filtres[0][0])):
                filtres[k][i][j] *= images[k][i][j]
    return filtres

def reverse_convolution(images, filtres, couche_active, classe, image_platte, lr=0.1):
    tableau_filtres = couche_convolutive_reverse(filtres, images, classe, image_platte)
    # tableau_filtres = []
    # tableau_filtres.append(couche_convolutive_reverse(images[2], image_platte))
    # tableau_filtres.append(couche_convolutive_reverse(images[1]))
    # tableau_filtres.append(couche_convolutive_reverse(images[0]))

    #mise_a_jour
    it1 = len(tableau_filtres); it2 = len(tableau_filtres[0]); it3 = len(tableau_filtres[0][0])
    for k in range(it1):
        for i in range(it2):
            for j in range(it3):
                filtres[k][i][j] -= images[2][k][i][j]*lr
    return filtres

def transpose(X):
    result = []
    for j in range(len(X[0])):
        mat = []
        for i in range(len(X)):
            mat.append(X[i][j])
        result.append(mat)
    return result

def couche_dense1(poids, image_applatie, tableau_biais=0):
    resultat = []
    som = 0
    for i in range(len(poids)):
        for j in range(len(poids[0])):
            som += poids[i][j]*image_applatie[j]
        resultat.append(som+tableau_biais)
    return resultat

def couche_dense2(image_applatie, poids, tableau_biais=0):
    resultat = []
    som = 0
    for i in range(len(poids[0])):
        for j in range(len(poids)):
            som += poids[j][i]*image_applatie[j]
        resultat.append(som+tableau_biais)
    return resultat

def activation_sigmoide(X):
    A=[]
    for x in X:
        A.append(1/(1+Math.exp(-1*x)))
    return A

def initialisation_matrices(nb, nb_lignes, nb_colonnes, type):
    sortie = []
    for k in range(nb):
        resultat = [[0]*nb_colonnes for i in range(nb_lignes)]
        for i in range(nb_lignes):
            for j in range(nb_colonnes):
                if type == 'float':
                    resultat[i][j] = random()/10000
                else:
                    resultat[i][j] = round(random()*10 +1)-5
        if nb == 1:
            return resultat
        else:
            sortie.append(resultat)

    return sortie

def calcul_lineaire(x, a_y):
    l1 = len(x); l2 = len(a_y)
    resultat = [[0]*l2 for i in range(l1)]
    temp = []; temp.append(x)
    x = transpose(temp)
    for i in range(len(x)):
        for j in range(len(a_y)):
            resultat[i][j] = x[i][0]*a_y[j]
    return resultat


def gradients(sortie_activation, sortie_model, classe):
    act = sortie_activation.copy()
    for i in range(len(act)):
        act[i] -= classe

    dW = calcul_lineaire(sortie_model, act)
    
    s = sum(act)
    dB = s/len(act)

    return dW, dB
    
def mise_a_jour(poids, biais, dW, dB, lr):
    # if len(poids) == 27:
    #     poids = transpose(poids)
    for i in range(len(dW)):
      
        for j in range(len(dW[0])):
            poids[i][j] -= dW[i][j]*lr
    biais -= dB*lr
    
    return poids, biais

def initialisation():
    w1 = initialisation_matrices(1, 10, 27, 'float')
    w2 = initialisation_matrices(1, 10, 2, 'float')
    w3 = initialisation_matrices(1, 2, 1, 'float')
    poids = [w1, w2, w3]
    biais = [random(), random(), random()]
    filtres = initialisation_matrices_f()
    # filtres2 = initialisation_matrices(3, 9, 9, 'int')
    # filtres3 = initialisation_matrices(3, 9, 9, 'int')
    # filtres = [filtres1, filtres2, filtres3]
    return poids, biais, filtres

def foward_propagation(image, tableau_filtres, tableau_poids ,tableau_biais, nb_conv=3):
    images_sorties = []
    
    for i in range(nb_conv):
        images_sorties.append(image)
        images_convoluees = convolution(image, tableau_filtres)
        images_activees = activation_relu(images_convoluees)
        images_reduites = maxPooling(images_activees)
        image = images_reduites

    images_applaties = applatissement(image)
    couche_dense_1 = couche_dense1(tableau_poids[0], images_applaties, tableau_biais[0])
    activation_couche1 = activation_sigmoide(couche_dense_1)
    couche_dense_2 = couche_dense2( activation_couche1, tableau_poids[1], tableau_biais[1])
    activation_couche2 = activation_sigmoide(couche_dense_2)
    couche_dense_3 = couche_dense2( activation_couche2, tableau_poids[2], tableau_biais[2])
    activation_couche3 = activation_sigmoide(couche_dense_3)

    return activation_couche3, couche_dense_3, activation_couche2, couche_dense_2, activation_couche1, couche_dense_1, images_sorties, images_applaties


def backward_propagation(activation_couche, couche_model, poids, biais, classe, lr=0.1):
    dpoids, dbiais = gradients(activation_couche, couche_model, classe)
    sortie_poids, sortie_biais = mise_a_jour(poids, biais, dpoids, dbiais, lr)
    return sortie_poids, sortie_biais 

from tqdm import tqdm

def entrainement(ensemble_images, classe, poids=[], biais=[], filtres=[], epoch=10):
    #ici nous entrenons les images de tailles 64x64
    if poids == [] and biais == [] and filtres == []:
        poids, biais, filtres = initialisation()

    for image in ensemble_images:
        for i in tqdm(range(epoch)):
            temp = []
            temp.append(image)
            couche_active3, couche_dense3, couche_active2, couche_dense2, couche_active1, couche_dense1, images, image_platte = foward_propagation(temp, filtres, poids, biais)
            poids[2], biais[2] = backward_propagation(couche_active3, couche_dense3, poids[2], biais[2], classe, 0.12)
            poids[1], biais[1] = backward_propagation(couche_active2, couche_dense2, poids[1], biais[1], classe, 0.12)
            poids[0], biais[0] = backward_propagation(couche_active1, couche_dense1, poids[0], biais[0], classe, 0.12)
            filtres = reverse_convolution(images, filtres, couche_active2, classe, image_platte, 0.12)

            with open('/media/brunel/ubuntu_xtend2/Neural-Network-master/filtre.csv', 'w') as file:
                for row in filtres:
                    file.write(' '.join(str(x) for x in row) + '\n')
            with open('/media/brunel/ubuntu_xtend2/Neural-Network-master/biais.csv', 'w') as file:
                for row in biais:
                    file.write(str(row )+ '\n')
            with open('/media/brunel/ubuntu_xtend2/Neural-Network-master/poids.csv', 'w') as file:
                for row in poids:
                    file.write(' '.join(str(x) for x in row) + '\n')

        return poids, biais, filtres

import csv
import numpy as np
from PIL import Image

def upload(fichier):
    lecture = csv.reader(open('/home/brunel/Downloads/Neural-Network-master/'+fichier+'.csv', 'r'), delimiter='\n')
    x = list(lecture)
    t = []
    for i in x:
        for j in range(len(i)):
            if fichier == 'biais':
                t.append(float(i[j]))
            else:
                t.append(i[j])
    return t

def prediction(img, poids=[], biais=[], filtre=[]):
    # poids = upload('poids')
    # biais  = upload('biais')
    # filtre  = upload('filtre')

    img = Image.open(img)
    img = img.resize((64, 64))
    img = img.convert("L")
    img = np.asarray(img)
    t = img.copy()
    for i in range(64):
        for j in range(64):
            t[i][j] /= 10000

    couche_active3, couche_dense3, couche_active2, couche_dense2, couche_active1, couche_dense1, images, image_platte = foward_propagation([t], filtre, poids, biais)
    
    if couche_active3[0] >0.55 :

        return "Chat "+str((couche_active3[0]))#+"%"
    elif couche_active3[0] <0.45 :
        return "Chien "+str((couche_active3[0]))#+"%"
    else:
        return "Autre "+str((couche_active3[0]))#+"%"






















def initialisation_matrices_f():
    filtres = [
    [[-1, -1, -1, -1, -1, -1, -1, -1, -1], 
     [-1, 1, 1, 1, 1, 1, 1, 1, -1 ], 
     [-1, 1, 2, 2, 2, 2, 2, 1, -1 ], 
     [-1, 1, 2, 3, 3, 3, 2, 1, -1 ], 
     [-1, 1, 2, 3, 4, 3, 2, 1, -1 ], 
     [-1, 1, 2, 3, 3, 3, 2, 1, -1 ], 
     [-1, 1, 2, 2, 2, 2, 2, 1, -1 ], 
     [-1, 1, 1, 1, 1, 1, 1, 1, -1 ], 
     [-1, -1, -1, -1, -1, -1, -1 , -1, -1]],

    [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
     [1, 3, 3, 3, 3, 3, 3, 3, 1], 
     [1, 3, 3, 3, 3, 3, 3, 3, 1], 
     [1, 2, 2, 2, 2, 2, 2, 2, 1], 
     [1, 2, 2, 2, 2, 2, 2, 2, 1], 
     [1, 1, 1, 1, 1, 1, 1, 1, 1], 
     [0, 1, 1, 1, 1, 1, 1, 1, 0], 
     [0, 0, 0, 0, 0, 0, 0, 0, 0], 
     [0, -1, -1, -1, -1, -1, -1, -1, 0]],

    [[-5, -5, -5, -5, -5, -5, -5, -5, -5], 
     [-5, 1, 1, 1, 1, 1, 1, 1, -5], 
     [-5, 1, 1, 1, 1, 1, 1, 1, -5], 
     [-5, 2, 2, 2, 2, 2, 2, 2, -5], 
     [-5, 2, 2, 2, 2, 2, 2, 2, -5], 
     [-5, 3, 3, 3, 3, 3, 3, 3, -5], 
     [-5, 3, 3, 3, 3, 3, 3, 3, -5], 
     [-5, 0, 0, 0, 0, 0, 0, 0, -5], 
     [-5, -5, -5, -5, -5, -5, -5, -5, -5]]
    ]
    return filtres