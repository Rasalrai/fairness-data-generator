import time
import os
import sys
import math
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def the_ratio(n, k):
    
    m = 1
    for i in range(1, n):
        m = m * (k + i)
    m = round(m / math.factorial(n - 1))
    print(m)
    return m


def genset_k_do_increment(k, X):
    
    a = np.shape(X)
    lastNZ = np.amax(np.matmul((X>0), np.diag(range(1, k+1))), axis=1)
    b = np.shape(lastNZ)
    k1lastNZ = k+1-lastNZ
    Y = np.repeat(X,k1lastNZ, axis=0)
    c = np.shape(Y)
    idxY = 1
    for q in range(1,a[0]+1):
        cs = lastNZ[q-1]
        for c in range(cs,k+1):
            Y[idxY+c-cs-1,c-1] += 1
        idxY = idxY + k+1-cs
    return Y


def genset_k_by_inc(n, k):

    m = the_ratio(n, k)

    X = np.zeros((m, n), dtype=np.int8, order='C')
    for i in range(0, n):
        X[i][i] = 1
    sm = 1 
    mX = the_ratio(n, sm)
    while (sm < k):
        tm = time.time()
        sm = sm + 1
        mX1 = the_ratio(n, sm)
        X[0:mX1-1, :] = genset_k_do_increment(n, X[0:mX - 1, :])
        mX = mX1
        print("iteration:%i -- %.2f [s]" % (k - sm, time.time() - tm))
        
    X[-1,-1] = X[0,0]
    return X


def generate_dataset(n, k):
    start_time = time.time()
    print("Generating simplex data",end='')
    print('\n')
    X = genset_k_by_inc(n, k)
    print(" -- Done")

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)
    print("GenSimpBIN: %.2f [s]" % (time.time() - start_time))
    return X


def save_bin_dataset(X, fname):
    start_time = time.time()
    print("Saving BIN file: %s" % fname, end='')
    with open(fname, 'wb') as f:
        pickle.dump(X, f)
    print("SaveBIN: %.2f [s]" % (time.time() - start_time))
    print(" -- Done")


def save_txt_dataset(X, fname):
    start_time = time.time()
    print("Saving TXT file: %s" % fname, end='')
    np.savetxt(fname, X, fmt='%i')
    print(" -- Done")
    print("SaveTXT: %.2f [s]" % (time.time() - start_time))


def load_bin_dataset(fname):

    print("Loading BIN file: %s" % fname, end='')
    with open(fname, 'rb') as f:
        X = pickle.load(f)
    print(" -- Done")

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)

    return X


def load_txt_dataset(fname):

    print("Loading TXT file: %s" % fname, end='')
    X = np.loadtxt(fname, dtype=np.int8)
    print(" -- Done")

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)

    return X


def get_group_ratio(df):
    df['gr'] = (df.f_tp + df.f_fp + df.f_tn + df.f_fn) / (df.m_tp + df.m_fp + df.m_tn + df.m_fn)
    return df


def get_imbalance_ratio(df):
    df['ir'] =  (df.f_tp + df.f_fp + df.m_tp + df.m_fp) / (df.f_tn + df.f_fn + df.m_tn + df.m_fn)
    return df


def get_class_minority_ratio(df):
    df['mr_class'] = (df.f_tp + df.f_fp + df.m_tp + df.m_fp) / ((df.f_tp + df.f_fp + df.m_tp + df.m_fp) + (df.f_tn + df.f_fn + df.m_tn + df.m_fn))
    return df


def get_group_minority_ratio(df):
    df['mr_group'] = (df.f_tp + df.f_fp + df.f_tn + df.f_fn) / ((df.f_tp + df.f_fp + df.f_tn + df.f_fn) + (df.m_tp + df.m_fp + df.m_tn + df.m_fn))
    return df


def get_true_pos_rate_ratio(df):
    df['tpr_ratio'] = (df.f_tp/(df.f_tp + df.f_fn)) / (df.m_tp/(df.m_tp + df.m_fn))
    return df


def get_men_true_pos_rate(df):
    df['m_tpr'] = df.m_tp/(df.m_tp + df.m_fn)
    return df


def get_females_true_pos_rate(df):
    df['f_tpr'] = df.f_tp/(df.f_tp + df.f_fn)
    return df


def get_true_pos_rate_diff(df):
    df['tpr_diff'] = df.f_tpr - df.m_tpr
    return df


def create_heatmap(df, fair_measure):
    df1 = df.groupby(['ir', fair_measure]).size().reset_index(name='counts')
    heatm = sns.heatmap(df1.pivot('ir', fair_measure, values='counts'), cmap="PiYG", annot=False)
    plt.title(str(fair_measure))
    #print(df1)
    #plt.show()
    fig = heatm.get_figure()
    fig.savefig(f"plots/heatmap_{fair_measure}.png") 
    

def create_histogram(df, ir_selected, fair_measure):
    df1 = df[df["ir"] == ir_selected]
    df1 = df1[[fair_measure]]
    hist = df1.hist(bins=100)
    plt.title(str(fair_measure) + ' for ir = ' + str(ir_selected))
    fig = hist[0][0].get_figure()
    fig.savefig(f"plots/histogram_{fair_measure}_{ir_selected}.png")
    #plt.show()
    
        
    
def create_dataframe(nparray, k):
    
    df = pd.DataFrame(nparray, columns=['m_tp', 'm_fp', 'm_tn', 'm_fn', 'f_tp', 'f_fp', 'f_tn', 'f_fn'])
    
    get_group_ratio(df)
    get_imbalance_ratio(df)
    get_class_minority_ratio(df)
    get_group_minority_ratio(df)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # normalize gr & ir
    # df.iloc[:,8:10] = df.iloc[:,0:-1].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    
    get_men_true_pos_rate(df)
    get_females_true_pos_rate(df)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # calculate fairness measures
    # equal opportunity ratio TP/(TP+FN)
    get_true_pos_rate_ratio(df)
    # equal opportunity difference
    get_true_pos_rate_diff(df)
    
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.replace(np.NaN, 0, inplace=True)
    
    # normalize fairness measure 
    # df.iloc[:,14] = df.iloc[:,0:-1].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    
    print(df)
    
    # 1 TODO: Automatyzacja robienia histogramów 
    # 2 TODO: Sprawdzenie kodu
    # 3 TODO: Etykietowanie osi
    # 4 TODO: Co z tymi zerowymi wartościami?
    # 5 TODO: Więcej miar
    
    return df


if __name__ == '__main__':

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        k = int(sys.argv[1])
        print('Comand line params: ', end='')
    else:
        # k - size of set; n -  error matrix combinations
        # k should be a multiple of n, (otherwise a potentially incomplete data set is generated)
        n = 8
        k = n * 2
        print('Default params: ', end='')
    print('n=%i, k=%i'%(n,k))

    prog_start_time = time.time()
    bin_fname = "Set(%02i,%02i).bin" % (n, k)

    # Data generating
    # X = generate_dataset(n, k)
    
    # Data saving - bin
    # save_bin_dataset(X, bin_fname)

    # Data loading - bin
    X = load_bin_dataset(bin_fname)
    
    # Data saving - txt
    # txt_fname = "Set(%02i,%02i).txt" % (n, k)
    # save_txt_dataset(X, txt_fname)
    
    df = create_dataframe(X, k)
    #create_heatmap(df, 'tpr_diff')
    #create_histogram(df, 15.0, 'tpr_diff')
    
    fm_list = ['tpr_ratio', 'tpr_diff']
    ir_selected_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    for fm in fm_list:
        create_heatmap(df, fm)
        for ir_selected in ir_selected_list:
            create_histogram(df, ir_selected, fm)
        
    print("Total time: %.2f [s]" % (time.time() - prog_start_time))

