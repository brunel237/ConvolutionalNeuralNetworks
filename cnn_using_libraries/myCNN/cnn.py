#Etant donne un dataset de DataSet(X, y) ; x1,x2 E X qui sont les caracteristiques a etudier p.ex
# les y sont les entites dans le dataset p.ex y=1 si c'est le chat et y=0 si c'est le chien

#Considerons que nous avons un dataset de DataSet(X,y) ;

import math as Math
import matplotlib.pyplot as plt
#import data as Data

def initialisation(X):
    pass


def train(X, W, B, F, y, learning_rate=0.2, epoch=3):
    cost = []
    for i in range(epoch):
        A = modeling(X, F, W, B, 1)
        print(cost_function(A, y))
        dW, dF, db = gradients(A, W, y)
        W, F, b = update(W, B, dW, dF, F, db, learning_rate)
    
    return W, F, b

def convolution(image, filtre, padding=0, stride=1):
    nb_fil=len(filtre)
    output_h = (len(image[0])-len(filtre[0][0])+padding)/stride  +1
    output_w = (len(image[0][0])-len(filtre[0][0][0])+padding)/stride  +1
    output = []

    if padding:
        pass

    while (nb_fil < len(filtre)):
        for K in range(len(image)):
            matrice = [] #= [None]*mat_len
            for i in range(len(image[0])-len(filtre[nb_fil][0])+stride):
                temp=[]
                for j in range(len(image[0][0])-len(filtre[nb_fil][0][0])+stride):
                    sum = 0
                    #for k in range(nblig):
                    kernel = [image[K][i][j:len(filtre[0][0][0])+j],
                              image[K][i+1][j:len(filtre[0][0][0])+j]
                            ]
                    for m in range(len(filtre[0][0])):
                        for n in range(len(filtre[0][0][0])):
                            sum+= filtre[nb_fil][K][m][n]*kernel[m][n]
                    temp.append(sum)
                matrice.append(temp)
            output.append(matrice)
        nb_fil+=1
        #print(output)

    
    return output

def maxPooling(image, nblig, nbcol, window=2):
    output = []
    k= 0

    while k < len(image):
        pooled = []; temp = []
        for i in range(0, nblig+1, window):
            for j in range(0, nbcol, window):
                fil = [image[k][i][j:2+j],
                       image[k][i+1][j:2+j],
                      ]
                max = fil[0][0]
                for m in range(len(fil)):
                    for n in range(len(fil[0])):
                        if fil[m][n] > max:
                            max = fil[m][n]
                temp.append(max)
            pooled.append(temp)
            temp=[]
        output.append(pooled)
        k+=1

    return output


def actRelu(X):
    for k in range(len(X)):
        for i in range(len(X[k])):
            for j in range(len(X[k][0])):
                if X[k][i][j] < 0:
                    X[k][i][j] = 0
    return X

def flattern(X):
    result = []; temp=[]
    for k in range(len(X)):
        for i in range(len(X[k])):
            for j in range(len(X[k][0])):
                temp.append(X[k][i][j])
        result.append(temp)
        temp = []
    return result

def modeling(X, F, W, b, nc=1):
    Z = []
    for i in range(nc):
        C = convolution(X, F, len(F))
        R = actRelu(C)
        P = maxPooling(R, 2, 2, 2)
        P = actRelu(P)
        X = P
    V = flattern(X)
    Z = linearity(W, V, b)
    
    return sigmoid_activation(Z)

def prediction(X,W,F,b):
    m = modeling(X, F, W, b)
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] <= 0.5:
                return 0
    return 1

def cost_function(A, y):
    sum=0; cost=[]
    for i in range(len(A)):
        for j in range(len(A[0])):
            sum += y*Math.log(A[i][j]) + (1-y)*Math.log(1-A[i][j])
        cost.append(sum*-1/len(A[0]))
        sum=0
    return cost


def gradients(A, W, y):
    T=[]
    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] -= y
    for i in range(len(A)):
        T.append(transpose(W[i]))
    
    temp = []
    for i in range(len(W)):
        temp.append(0)
    dW = linearity(T, A, temp)
    
    for i in range(len(dW)):
        for j in range(len(dW)):
            dW[i][j] /= len(A)
    
    sum =0
    for i in range(len(A)):
        for j in range(len(A[0])):
            sum += A[i][j]
    
    dF = linearity(W, A, temp)
    db=[]
    sum=0
    for x in A:
        for i in range(len(x)):
            sum += x[i]
        db.append(sum/len(A))
        sum=0
    
    return dW, dF, db


def update(W, b, dW, dF, F, db, learning_rate):
    for k in range(len(W)):
        for i in range(len(W[k])):
            for j in range(len(W[k][0])):
                W[k][i][j] -= learning_rate*dW[k][i]
    
    for i in range(len(b)):
        b[i] -= learning_rate*db[i]
    
    for k in range(len(F)):
        for i in range(len(F[k])):
            for j in range(len(F[k][0])):
                for l in range(len(F[k][0][0])):
                    F[k][i][j][l] -= learning_rate*dF[k][i]
    
    return W, F, b



def sigmoid_activation(X):
    A=[]; temp = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            temp.append(1/(1+Math.exp(-1*X[i][j])))
        A.append(temp)
        temp=[]
    return A

def transpose(X):
    result = []
    for j in range(len(X[0])):
        mat = []
        for i in range(len(X)):
            mat.append(X[i][j])
        result.append(mat)
    return result

def linearity(X, W, b=0):
    Z=[]; output = []
    for k in range(len(X)):
        for i in range(len(X[k])):
            product = 0
            for j in range(len(X[k][0])):
                product += X[k][i][j]*W[k][j]
            Z.append(product+b[i])
        output.append(Z)
        Z =[]
    return output

