import numpy as np

def readPgm(name):
    img=open(name)
    format=img.readline()
    line=img.readline()
    while line[0] == '#' : line=img.readline()
    (width,height) = [int(i) for i in line.split()]
    valmax=img.readline()
    raster = []
    for i in range(width):
        line = img.readline().split()
        for j in range(height):
            raster.append(int(line[j]))
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

def readCoeffCNN(name):
    CNNfile = open("CNN_coeff_3x3.txt")
    line = CNNfile.readline()
    linesplit = line.split()
    while linesplit[1] != (name + "/biases"):
        linesplit = CNNfile.readline()
    biases = []
    linesplit = CNNfile.readline().split()
    while linesplit[0] != "tensor_name:":
        for coeff in linesplit:
            biases.append(coeff)
        linesplit = CNNfile.readline().split()
    biases[0] = biases[0][1:]
    biases[-1] = biases[-1][:-2]
    for i in range(len(biases)):
        biases[i] = float(biases[i])

    line = CNNfile.readline()
    tempFile = open("temptensor.npy",'w')
    while line[0] != "t":
        tempFile.write(line)
        line = CNNfile.readline()
    tempFile.close()
    tempFile = open("temptensor.npy",'r+')
    weights = np.array([np.array(i) for i in tempFile.readlines()])
    #weights = [i for i in tempFile.readlines()]
    tempFile.close()
    print(weights)
    return biases, weights
    '''i = -1
    while line[0] != "t":
        linesplit = line.split()
        if len(linesplit) > 4:
            if linesplit == "[[[[" or linesplit == "[[[":
                i++
            elif linesplit == "[[":
                de
            elif linesplit == "[":
    regexp = r"(\d+)\s+(...)"
    weights = np.fromregex("temptensor.txt", regexp, dtype=None)
    tempFile.close()
    return biases, weights'''

#writePgm("cifar10_voilier_resize.pgm")
biases, weights = readCoeffCNN("conv1")
print(weights)
print(biases)
#TODO : lecteur du fichier CNN3x3
