import numpy as np
import os
import sys
import pdb

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

def writePgm(name):
    pgm = readPgm("cifar10_voilier_bin.pgm")
    pgm=cutPgm3224(pgm)
    img=open(name, 'w')
    img.write("P2\n")
    img.write("24 24\n")
    img.write("255\n")
    for i in pgm:
        img.write(str(i)+" ")
    img.close()

def RELU(a):
    return max(0,a)

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
    print(K.shape)
    print(K.shape[2])
    S = np.zeros([I.shape[0], I.shape[1], len(b)])
    for c in range(len(b)):
        for j in range(I.shape[1]):
            for i in range(I.shape[0]):
                for l in range(K.shape[2]):
                    for m in range(-1, K.shape[0]-1):
                        for n in range(-1, K.shape[1]-1):
                            #print("m", m, "n", n, "i+m", i+m, "j+n", j+n)
                            if 0 <= i+m < I.shape[0] and 0 <= j+n < I.shape[1]:
                                #print("S", S[i][j][c])
                                #if S[i][j][c] > 10**10:
                                    #pdb.set_trace()
                                    #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                                    #print("i",i)
                                    #print("j",j)
                                    #print("c",c)
                                S[i][j][c] += I[i+m][j+n][l]*K[m][n][l][c]
                #print("S(",i,",",j,",",c,")=", S[i][j][c])
                S[i][j][c] = RELU(S[i][j][c] + b[c])
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

def whatIs(name):
    f = open("cifar10_data/cifar-10-batches-bin/batches.meta.txt")
    K, b = readCoeffCNN("conv1", 3)
    K2, b2 = readCoeffCNN("conv2", 64)
    K3, b3 = readCoeffCNN("conv3", 32)
    l = readMatrixCNN("local3")
    num = np.reshape(MaxPool(conv(MaxPool(conv(MaxPool(conv(readPpm(sys.argv[1]), K, b)), K2, b2)), K3, b3)), 180) @ l
    print(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    print(num)
    for i in range(len(num)):
        whatitis = f.readline()
        if num[i] == max(num):
            return i, whatitis, num[i]
    return "WTF"


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
"""


print(whatIs(sys.argv[1]))
"""
V = readPpm(sys.argv[1])
print("VVVVVVVVVVVVVV---------------------------------------------------------")
print(V)
K, b = readCoeffCNN("conv1", 3)
print("KKKKKKKKKKKKKKK---------------------------------------------------------")
print(K)
C = conv(V, K, b)
print("CCCCCCCCCCCCCC---------------------------------------------------------")
print(C)
M = MaxPool(C, 12, 12)
print("MMMMMMMMMMMMMM---------------------------------------------------------")
print(M)
K2, b2 = readCoeffCNN("conv2", 64)
print("KKKKKKKKKKKKKKK2222222222222222222---------------------------------------------------------")
print(K2)
C2 = conv(M, K2, b2)
print("CCCCCCCCCCCCCCC222222222222222222---------------------------------------------------------")
print(C2)
M2 = MaxPool(C2, 6, 6)
print("MMMMMMMMMMMMMMM222222222222222222---------------------------------------------------------")
print(M2)
K3, b3 = readCoeffCNN("conv3", 32)
print("KKKKKKKKKKKKKK33333333333333333---------------------------------------------------------")
print(K3)
C3 = conv(M2, K3, b3)
print("CCCCCCCCCCCCC33333333333333333--------------------------------------------------------")
print(C3)
np.shape(C3)
M3 = MaxPool(C3, 3, 3)
print("MMMMMMMMMMMMM-33333333333333333---------------------------------------------------------")
print(M3)
M3 = np.reshape(M3, 180)
print("MMRESHHHHH---------------------------------------------------------")
print(M3)
l = readMatrixCNN("local3")
l = np.transpose(l)
print("LLLLLLLLLL---------------------------------------------------------")
print(l)
res = l @ M3
print(res)

A = np.ones([24,24,64])
B = np.ones([3,3,64,32])
C = np.ones([32])
print("A", A, "\nB", B, "\nC", C)
D = conv(A, B, C)
print("\nD", np.shape(D))
print(D)
"""


#print(conv(readPpm("cifar10_voilier_bin.ppm"), K, b))
#print(readPpm("cifar10_voilier_bin.ppm"))
#print(listtoarray(cutPgm3224(readPgm("cifar10_voilier_bin.pgm"))))
#writePgm("cifar10_voilier_resize.pgm")
#biases, weights = readCoeffCNN("conv1")
