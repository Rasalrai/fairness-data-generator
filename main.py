import time
import sys
import math
import numpy
import pickle
import pandas as pd


def the_ratio(n, k):
    m = 1
    for i in range(1, n):
        m = m * (k + i)
    m = round(m / math.factorial(n - 1))
    print(m)
    return m


def genSet_k_do_increment (k,X):
    #print(X)
    a=numpy.shape(X)
    lastNZ = numpy.amax(numpy.matmul((X>0),numpy.diag(range(1,k+1))),axis=1)
    #print(lastNZ)
    b=numpy.shape(lastNZ)
    k1lastNZ = k+1-lastNZ
    Y = numpy.repeat(X,k1lastNZ,axis=0)
    #print(Y)
    c=numpy.shape(Y)
    idxY = 1
    for q in range(1,a[0]+1):
        cs = lastNZ[q-1]
        for c in range(cs,k+1):
            Y[idxY+c-cs-1,c-1] += 1
        idxY = idxY + k+1-cs
    return Y


def genSet_k_by_inc (n, k):

    m = the_ratio(n, k)

    X = numpy.zeros((m, n), dtype=numpy.int8, order='C')
    #print('Size:',numpy.shape(X)[0])
    for i in range(0, n):
        X[i][i] = 1

    sm = 1 # sum(X(1,:))
    mX = the_ratio(n, sm)
    while (sm < k):
        tm = time.time()
        sm = sm + 1
        mX1 = the_ratio(n, sm)
        X[0:mX1-1,:] = genSet_k_do_increment(n, X[0:mX - 1, :])
        mX = mX1
        print("iteration:%i -- %.2f [s]" % (k - sm, time.time() - tm))

    X[-1,-1] = X[0,0]
    return X


def GenerateSimpTheDataSet(n, k):

    print("Generating simplex data",end='')
    print('\n')
    X = genSet_k_by_inc(n, k)
    print(" -- Done")

    mn = numpy.shape(X)
    print(numpy.sum(X[0, :]))
    print(mn)

    return X


def SaveBINTheDataSet(X, fname):

    print("Saving BIN file: %s" % fname, end='')
    #numpy.save(fname, X, allow_pickle=True)
    with open(fname, 'wb') as f:
        pickle.dump(X, f)
    print(" -- Done")


def SaveTXTTheDataSet(X, fname):

    print("Saving TXT file: %s" % fname, end='')
    numpy.savetxt(fname, X, fmt='%i')
    print(" -- Done")


def LoadBINTheDataSet(fname):

    print("Loading BIN file: %s" % fname, end='')
    #X = numpy.load(fname,allow_pickle=True)
    with open(fname, 'rb') as f:
        X = pickle.load(f)
    print(" -- Done")

    mn = numpy.shape(X)
    print(numpy.sum(X[0, :]))
    print(mn)

    return X


def LoadTXTTheDataSet(fname):

    print("Loading TXT file: %s" % fname, end='')
    X = numpy.loadtxt(fname, dtype=numpy.int8)
    print(" -- Done")

    mn = numpy.shape(X)
    print(numpy.sum(X[0, :]))
    print(mn)

    return X

def read_into_dataframe(nparray):
    structured_data = pd.DataFrame(nparray, columns=['m_tp', 'm_fp', 'm_tn', 'm_fn', 'f_tp', 'f_fp', 'f_tn', 'f_fn'])
    print(structured_data)
    return structured_data


######################################


if __name__ == '__main__':

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        k = int(sys.argv[1])
        print('Comand line params: ',end='')
    else:
        n = 8
        # k should be a multiple of n
        # (otherwise a potentially incomplete data set is generated)
        k = n*2
        print('Default params: ',end='')
    print('n=%i, k=%i'%(n,k))

    prog_start_time = time.time()

    # Generowanie danych
    #
    start_time = time.time()
    X = GenerateSimpTheDataSet(n, k)
    print("GenSimpBIN: %.2f [s]" % (time.time() - start_time))
    
    read_into_dataframe(X)

    # # Zapisywanie danych (wersja binarna)
    # # (dane zapisane w ponizszy sposob odczytuje procedura LoadBINTheDataSet))
    # #
    # bin_fname = "Set(%02i,%02i).bin" % (n, k)
    # start_time = time.time()
    # SaveBINTheDataSet(X, bin_fname)
    # print("SaveBIN: %.2f [s]" % (time.time() - start_time))

    # # Zapisywanie danych (wersja tekstowa)
    # # -- alternatywa do wersji binarnej
    # # -- NIE UZYWAC dla duzych zbiorow danych (jest zaskakujaco powolna!)
    # # (dane zapisane w ponizszy sposob odczytuje procedura LoadTXTTheDataSet))
    # #
    # txt_fname = "Set(%02i,%02i).txt" % (n, k)
    # start_time = time.time()
    # SaveTXTTheDataSet(X, txt_fname)
    # print("SaveTXT: %.2f [s]" % (time.time() - start_time))

    # print("Total time: %.2f [s]" % (time.time() - prog_start_time))

    # # k - rozmiar zbioru, 8 - dwie macierze pom

