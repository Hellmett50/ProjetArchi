import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *   ###################



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


def printtrace(pathconv, C):
    for c in range(C.shape[2]):
        writePgm(pathconv+str(c), C, c)
        #writePgm(pathmax+str(c), M, c)
    return 0

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
    linesplit = CNNfile.readline().replace("[","").replace("]","").split()#.replace("\n","")  ###################

    while linesplit[0] != "tensor_name:":
        for coeff in linesplit:
            biases.append(coeff)
        linesplit = CNNfile.readline().replace("[","").replace("]","").split()   ###################

    #del biases[0]  ###################
    biases[-1] = biases[-1][:-2]

    for i in range(len(biases)):
        biases[i] = float(biases[i])


    i = 0
    j = 0
    k = 0
    weights = np.zeros([3, 3, length, len(biases)])
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

def conv(I, K, b):
    S = np.zeros([I.shape[0], I.shape[1], len(b)])
    M = np.zeros([3, 3, len(b)])
    for c in range(len(b)):
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                for l in range(K.shape[2]):
                    for m in range(K.shape[0]):
                        for n in range(K.shape[1]):
                            if 0 < i+m <= I.shape[0] and 0 < j+n <= I.shape[1]:
                                S[i][j][c] += I[i+m-1][j+n-1][l]*K[m][n][l][c]
                S[i][j][c] = max(S[i][j][c] + b[c], 0)
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
                M[m][n][c] = max(S00, S01, S02, S10, S[2*m][2*n][c], S12, S20, S21, S22)
    return M

def whatIs(name):
    K, b = readCoeffCNN("conv1", 3)
    K2, b2 = readCoeffCNN("conv2", 64)
    K3, b3 = readCoeffCNN("conv3", 32)
    l = readMatrixCNN("local3")

    V = readPpm(name)


    ###################################### partie normalization de l'image ##########
    reshaped_image = V.reshape((-1,3))
    mean = np.mean(reshaped_image, axis=0)
    mean_array = np.ones(reshaped_image.shape)*mean
    sigma = np.var(reshaped_image, axis=0)
    sigma_array = np.ones(reshaped_image.shape)*sigma
    reshaped_image = (reshaped_image-mean_array)/np.maximum(np.sqrt(sigma_array),1/sqrt(reshaped_image.shape[0]))
    reshaped_image = reshaped_image.reshape((V.shape[0], V.shape[1],3))
    #################################################################################


    V=reshaped_image

    C = conv(V, K, b)
    M = MaxPool(C)

    C2 = conv(M, K2, b2)
    M2 = MaxPool(C2)

    C3 = conv(M2, K3, b3)
    M3 = MaxPool(C3)

    R = np.reshape(M3, 180)

    out = R @ l
    print(out)

    val = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    max_idx = int(np.where(out == np.amax(out))[0])  ###################
    print("\nthe max index is : ", max_idx)            ###################
    print("the label is : ", val[max_idx], "\n")           ###################

    return val[max_idx]



category = sys.argv[1][12:-4].split('_')[1]
print(category)
itis = whatIs(sys.argv[1])    ###################
print(category == itis)
#print(itis[0])               ###################
#print(itis[1])               ###################
trace = open("trace", 'a')
trace.write(category+"=="+itis+"="+str(category==itis)+'\n')
trace.close()
