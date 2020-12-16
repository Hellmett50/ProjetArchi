import tensorflow as tf
import numpy as np


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

inputshape = (1, 6, 6, 1)
input = tf.random.normal(inputshape)

#input = np.ones([6,6,1])
kernel = np.ones([3,3,1,2])
bias = np.array([2, 3])
C = conv(input[0], kernel, bias)
kernel = tf.constant_initializer(kernel)
bias = tf.constant_initializer(bias)
C2 = tf.keras.layers.Conv2D(2, (3, 3), activation='relu', padding='same', input_shape=input.shape[1:], use_bias=True, kernel_initializer=kernel, bias_initializer=bias)(input)

print(C)
print(C2)
print(C2 == C)
