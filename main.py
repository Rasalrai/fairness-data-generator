import time
import sys
import math
from unicodedata import category
import numpy
import pickle
import pandas as pd
import matplotlib as plt


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

def read_into_dataframe(nparray, k):
    df = pd.DataFrame(nparray, columns=['m_tp', 'm_fp', 'm_tn', 'm_fn', 'f_tp', 'f_fp', 'f_tn', 'f_fn']) 
    df['gr'] = (df.f_tp + df.f_fp + df.f_tn + df.f_fn) / (df.m_tp + df.m_fp + df.m_tn + df.m_fn)
    df['ir'] = (df.f_tn + df.f_fn + df.m_tn + df.m_fn) / (df.f_tp + df.f_fp + df.m_tp + df.m_fp) 
    
    #to zeros?
    #df.replace([numpy.inf, -numpy.inf], 0, inplace=True)
    
    df.replace([numpy.inf, -numpy.inf], 0, inplace=True)
    
    df.iloc[:,8:10] = df.iloc[:,0:-1].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    #equal opportunity ratio TP/(TP+FN)
    
    df['m_tpr'] = df.m_tp/(df.m_tp + df.m_fn)
    df['f_tpr'] = df.f_tp/(df.f_tp + df.f_fn)
    df.replace([numpy.inf, -numpy.inf], 0, inplace=True)
    df['tpr_ratio'] = (df.f_tp/(df.f_tp + df.f_fn)) / (df.m_tp/(df.m_tp + df.m_fn))
    df.replace([numpy.inf, -numpy.inf], 0, inplace=True)
    df.replace(numpy.NaN, 0, inplace=True)
    df.iloc[:,12] = df.iloc[:,0:-1].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    
    ###histogram logic
    df1 = df.groupby(['ir', 'tpr_ratio']).size().reset_index(name='counts')
    
    #pd.set_option('display.max_rows', None)
    
    print(df1)
    #check for ir in the middle
    #print(df.loc[9200,])
    #numpy.savetxt('df', df.values)

    
    return df


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
    # start_time = time.time()
    # X = GenerateSimpTheDataSet(n, k)
    # print("GenSimpBIN: %.2f [s]" % (time.time() - start_time))
    
    #read_into_dataframe(X, k)

    # Zapisywanie danych (wersja binarna)
    # (dane zapisane w ponizszy sposob odczytuje procedura LoadBINTheDataSet))
    #
    bin_fname = "Set(%02i,%02i).bin" % (n, k)
    X = LoadBINTheDataSet(bin_fname)
    read_into_dataframe(X, k)
    
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

