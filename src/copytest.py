import numpy as np
import os
import sys
import pdb
#import matplotlib.pyplot as plt
#import torch
import tensorflow as tf

def readPgm(name):
    img=open(name,'r')
    img.seek(0, os.SEEK_END)
    eof = img.tell()
    img.seek(os.SEEK_SET)

    format=img.readline()
    line=img.readline()
    while line[0] == '#' : line=img.readline()
    (width,height) = [int(i) for i in line.split()]
    valmax=img.readline()
    raster=[]
    while(img.tell() != eof):
        line=img.readline()
        for i in line.split():
            raster.append(int(i))
    return raster


def readPpm(name):
    img=open(name)
    format=img.readline()
    line=img.readline()
    while line[0] == '#' : line=img.readline()
    (width,height) = [int(i) for i in line.split()]
    valmax=img.readline()
    raster = np.empty([width-8, height-8, 3])
    for i in range(4, width-4):
        line = img.readline().split()
        for j in range(4, height-4):
            raster[i-4][j-4][0] = line[3*j]
            raster[i-4][j-4][1] = line[3*j+1]
            raster[i-4][j-4][2] = line[3*j+2]
    img.close()
    return raster

def cutPgm3224(pgmtab):
    newtab = []
    for i in range(len(pgmtab)):
        if 4*32 < i <= 28*32 and 4 < i%32 <= 28:
            newtab.append(pgmtab[i])
    return newtab

def writePgm(name, pgm, c):
    #pgm = readPgm("cifar10_test_bin.pgm")
    #pgm=cutPgm3224(pgm)
    img=open(name, 'w')
    img.write("P2\n")
    img.write(str(pgm.shape[0])+" "+str(pgm.shape[1])+"\n")
    img.write("255\n")
    for i in range(pgm.shape[0]):
        for j in range(pgm.shape[1]):
            img.write(str(int(pgm[i][j][c]))+" ")
    img.close()

def readCoeffCNN(name, length):
    CNNfile = open("CNN_coeff_3x3.txt")
    line = CNNfile.readline()
    linesplit = line.split()
    read = 1
    while read:
        if len(linesplit) == 2:
            if linesplit[1] == name + "/biases":
                read = 0
                break
        line = CNNfile.readline()
        linesplit = line.split()
    biases = []
    linesplit = CNNfile.readline().split()
    while linesplit[0] != "tensor_name:":
        for coeff in linesplit:
            biases.append(coeff)
        linesplit = CNNfile.readline().split()
    del biases[0]
    biases[-1] = biases[-1][:-2]
    for i in range(len(biases)):
        biases[i] = float(biases[i])

    i = 0
    j = 0
    k = 0
    weights = np.empty([3, 3, length, len(biases)])
    coeff = []
    line = CNNfile.readline()
    linesplit = line.split()[1:]
    while line[0] != "t":
        if line != '\n':
            if len(linesplit) > 4:
                weights[i][j][k] = coeff
                if linesplit[0] == "[[[":
                    i+=1
                    j=0
                    k=0
                elif linesplit[0] == "[[":
                    j+=1
                    k=0
                elif linesplit[0] == "[":
                    k+=1
                linesplit = linesplit[1:]
                coeff = list(map(float, linesplit))
            else:
                try:
                    float(linesplit[-1])
                except:
                    linesplit[-1] = linesplit[-1].replace(']', '')
                coeff = coeff + list(map(float, linesplit))
        line = CNNfile.readline()
        linesplit = line.split()
    CNNfile.close()
    weights[i][j][k] = coeff
    return weights, biases

def readMatrixCNN(name):
    CNNfile = open("CNN_coeff_3x3.txt")
    CNNfile.seek(0, os.SEEK_END)
    eof = CNNfile.tell()
    CNNfile.seek(os.SEEK_SET)

    read = 1
    while read:
        line = CNNfile.readline()
        linesplit = line.split()
        if len(linesplit) == 2:
            if linesplit[1] == name + "/weights":
                read = 0
    line = CNNfile.readline()
    linesplit = line.split()
    M = np.empty([180, 10])
    i = 0
    j = 0
    while linesplit != []:
        M[i][j] = linesplit[1]
        M[i][j+1] = linesplit[2]
        M[i][j+2] = linesplit[3]
        M[i][j+3] = linesplit[4]
        linesplit = CNNfile.readline().split()
        M[i][j+4] = linesplit[0]
        M[i][j+5] = linesplit[1]
        M[i][j+6] = linesplit[2]
        M[i][j+7] = linesplit[3]
        linesplit = CNNfile.readline().split()
        M[i][j+8] = linesplit[0]
        M[i][j+9] = linesplit[1].replace(']', '')
        linesplit = CNNfile.readline().split()
        i+=1
    return M

def listtoarray(list):
    i=0
    j=0
    list2d = np.empty([24, 24])
    for value in list:
        list2d[j][i] = value
        i+=1
        if i%24 == 0:
            j+=1
            i=0
    return list2d

def conv(I, K, b):
    #print(K.shape)
    #print(K.shape[2])
    S = np.zeros([I.shape[0], I.shape[1], len(b)])
    M = np.zeros([3, 3, len(b)])
    for c in range(len(b)):
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                for l in range(K.shape[2]):
                    for m in range(K.shape[0]):
                        for n in range(K.shape[1]):
                            #print("m", m, "n", n, "i+m", i+m, "j+n", j+n)
                            if 0 < i+m <= I.shape[0] and 0 < j+n <= I.shape[1]:
                                M[m][n][c] = K[m][n][l][c]
                                S[i][j][c] += I[i+m-1][j+n-1][l]*K[m][n][l][c]
                                #if S[i][j][c] > 10**10:
                                    #print("S", S[i][j][c])
                                    #print(S.shape)
                                    #pdb.set_trace()
                                    #S[i][j][c] = 255
                                    #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                                    #print("i",i)
                                    #print("j",j)
                                    #print("c",c)
                                #print("S(",i,",",j,",",c,") += I[", i+m-1,"][", j+n-1, "][", l, "]*K[", m, "][", n, "][", l, "][", c, "]")
                                #print(S[i][j][c], " += ", I[i+m-1][j+n-1][l], "*", K[m][n][l][c])
                #print("S(",i,",",j,",",c,")=", S[i][j][c])
                S[i][j][c] = max(S[i][j][c] + b[c], 0)
                #print(M[:][:][c], "M"+str(m)+str(n))
    return S

def MaxPool(S):
    M = np.zeros([S.shape[0]//2, S.shape[1]//2, S.shape[2]])
    for c in range(S.shape[2]):
        for m in range(0, M.shape[0]):
            for n in range(0, M.shape[1]):
                S00 = 0
                S01 = 0
                S02 = 0
                S10 = 0
                S12 = 0
                S20 = 0
                S21 = 0
                S22 = 0
                if m>0:
                    S01 = S[2*m-1][2*n][c]
                    if n>0:
                        S00 = S[2*m-1][2*n-1][c]
                    if n<S.shape[1]//2:
                        S02 = S[2*m-1][2*n+1][c]
                if m<S.shape[0]//2:
                    S21 = S[2*m+1][2*n][c]
                    if n>0:
                        S20 = S[2*m+1][2*n-1][c]
                    if n<S.shape[1]//2:
                        S22 = S[2*m+1][2*n+1][c]
                if n>0:
                    S10 = S[2*m][2*n-1][c]
                if n<S.shape[1]//2:
                    S12 = S[2*m][2*n+1][c]
                #print(np.array([[S00, S01, S02], [S10, S[2*m][2*n][c], S12], [S20, S21, S22]]))
                M[m][n][c] = max(S00, S01, S02, S10, S[2*m][2*n][c], S12, S20, S21, S22)
    return M

def printtrace(pathconv, C, pathmax, M):
    for c in range(C.shape[2]):
        writePgm(pathconv+str(c), C, c)
        writePgm(pathmax+str(c), M, c)


def whatIs(name):
    K, b = readCoeffCNN("conv1", 3)
    K2, b2 = readCoeffCNN("conv2", 64)
    K3, b3 = readCoeffCNN("conv3", 32)
    l = readMatrixCNN("local3")
    V = readPpm(name)
    #print(V)
    C = conv(V, K, b)
    #print(C)
    M = MaxPool(C)
    #print(M)
    printtrace("img/conv1/cifar10test", C, "img/max1/cifar10test", M)
    C2 = conv(M, K2, b2)
    #print(C2)
    M2 = MaxPool(C2)
    #print(M2)
    printtrace("img/conv2/cifar10test", C2, "img/max2/cifar10test", M2)
    C3 = conv(M2, K3, b3)
    #print(C3)
    M3 = MaxPool(C3)
    #print(M3)
    printtrace("img/conv3/cifar10test", C3, "img/max3/cifar10test", M3)
    R = np.reshape(M3, 180)
    #print(R)
    out = R @ l

    val = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    #print(val)
    #print(out)
    for i in range(len(out)):
        if out[i] == max(out):
            return out, val[i]
    return 0

def tf_whatIs(name):
    K, b = readCoeffCNN("conv1", 3)
    K2, b2 = readCoeffCNN("conv2", 64)
    K3, b3 = readCoeffCNN("conv3", 32)
    l = readMatrixCNN("local3")
    V = readPpm(name)

    K = tf.constant_initializer(K)
    b = tf.constant_initializer(b)
    K2 = tf.constant_initializer(K2)
    b2 = tf.constant_initializer(b2)
    K3 = tf.constant_initializer(K3)
    b3 = tf.constant_initializer(b3)
    maxpool2d = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides = (2, 2), padding = 'same')
    V = np.reshape(V, (1, 24, 24, 3))
    C = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=V.shape[1:], use_bias=True, kernel_initializer=K, bias_initializer=b)(V)
    print(C.shape)
    M = maxpool2d(C)
    print(M.shape)
    printtrace("img/conv1/cifar10test", C[0], "img/max1/cifar10test", M[0])
    C2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=M.shape[1:], use_bias=True, kernel_initializer=K2, bias_initializer=b2)(M)
    print(C2.shape)
    M2 = maxpool2d(C2)
    print(M2.shape)
    printtrace("img/conv2/cifar10test", C2[0], "img/max2/cifar10test", M2[0])
    M2 = np.reshape(M2, (1, 6, 6, 32))
    C3 = tf.keras.layers.Conv2D(20, (3, 3), activation='relu', padding='same', input_shape=M2.shape[1:], use_bias=True, kernel_initializer=K3, bias_initializer=b3)(M2)
    print(C3.shape)
    M3 = maxpool2d(C3)
    print(M3.shape)
    printtrace("img/conv3/cifar10test", C3[0], "img/max3/cifar10test", M3[0])
    R = np.reshape(M3, 180)
    #print(R)
    out = R @ l
    #out = tf.contrib.layers.fully_connected(R, 10, activation_fn=None, weights_initializer=tf.constant_initializer(l),)
    val = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(val)
    print(out)
    for i in range(len(out)):
        if out[i] == max(out):
            return out, val[i]
            #return val[i]

    #out = np.reshape(MaxPool(conv(MaxPool(conv(MaxPool(conv(readPpm(sys.argv[1]), K, b)), K2, b2)), K3, b3)), 180) @ l


    return 0


"""

I = np.array([[[1,160],[2,150],[3,140],[4,130]],
            [[5,120],[6,110],[7,100],[8,90]],
            [[9,80],[10,70],[11,60],[12,50]],
            [[13,40],[14,30],[15,20],[16,10]]])
K = np.ones([3, 3, 2, 4])
b = np.array([1, 2, 3, 4])
print("I", I)
print("I[0][1][0]", I[0][1][0])
print("I[1][0][0]", I[1][0][0])

M = MaxPool(I)
print("M", M)
C = conv(I, K, b)
print(C)


itis = whatIs(sys.argv[1])
#print(sys.argv[1][12:]+"="+itis)
tfitis = tf_whatIs(sys.argv[1])
print(itis, tfitis)
"""
K, b = readCoeffCNN("conv1", 3)
K2, b2 = readCoeffCNN("conv2", 64)
K3, b3 = readCoeffCNN("conv3", 32)

print("K")
print(K)
print("K2")
print(K2)
print("K3")
print(K3)
