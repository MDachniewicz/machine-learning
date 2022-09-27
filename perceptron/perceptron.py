# -*- coding: utf-8 -*-


import numpy as np

def perceptron(data, labels, params={}, hook=None):
    T=params.get('T',100)
    d, n = data.shape
    th = np.zeros((d,1))
    th0 = np.zeros(1)
    mistakes=0
    for t in range(T):
        for i in range(n):
            if labels[0,i]*(np.dot(th[:,0].T, data[:,i])+th0)<=0:
                th[:,0]=th[:,0]+labels[0,i]*data[:,i]
                th0=th0+labels[0,i]
                mistakes+=1
                if hook: hook((th, th0))
    print(mistakes)           
    return (th,th0)


def perceptron_origin(data, labels, params={}, hook=None):
    T=params.get('T',100)
    d, n = data.shape
    data=np.append(data, np.ones((n,1)).T, axis=0)
    
    th = np.zeros((d+1,1))
    #print(data)
    mistakes=0
    
    for t in range(T):
        for i in range(n):
            if labels[0,i]*(np.dot(th[:,0].T, data[:,i]))<=0:
                th[:,0]=th[:,0]+labels[0,i]*data[:,i]
                #print(data[:,i])
                mistakes+=1
                if hook: hook((th))
    print(mistakes)           
    return th

def averagedperceptron(data, labels, params={}, hook=None):
    T=params.get('T',100)
    d, n = data.shape
    th = np.zeros((d,1))
    th0 = np.zeros(1)
    ths = np.zeros((d,1))
    th0s = np.zeros(1)
    for t in range(T):
        for i in range(n):
            if labels[0,i]*(np.dot(th[:,0].T, data[:,i])+th0)<=0:
                th[:,0]=th[:,0]+labels[0,i]*data[:,i]
                th0=th0+labels[0,i]
                if hook: hook((th, th0))               
            ths+=th
            th0s+=th0
            

    return (ths/(n*T),th0s/(n*T))

def y(x, th, th0):
    return np.dot(np.transpose(th), x) + th0

def positive(x, th, th0):
    return np.sign(y(x, th, th0))

def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)
    
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0=learner(data_train, labels_train)
    d, n = data_test.shape
    scr=score(data_test, labels_test, th, th0)
    return scr/n

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    scr = 0
    for i in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test) 
        scr_i = eval_classifier(learner, data_train, labels_train, data_test, labels_test)
        
        scr+= scr_i
    
    return scr/it

def eval_learning_alg1(learner, data_gen, n_train, n_test, it): #training data == test data
    scr = 0
    for i in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test) 
        scr_i = eval_classifier(learner, data_train, labels_train, data_train, labels_train)
        
        scr+= scr_i
    
    return scr/it


def xval_learning_alg(learner, data, labels, k):
    data_i = np.array_split(data,k,axis=1)
    labels_i = np.array_split(labels,k,axis=1)
    scr=np.zeros(k)
    for j in range(k):
        d,n=data_i[j].shape
        D_minus_j = np.concatenate(list(data_i[n] for n in range(k) if n!=j), axis=1)
        labels_minus_j = np.concatenate(list(labels_i[n] for n in range(k) if n!=j), axis=1)
        th,th0 = learner(D_minus_j, labels_minus_j)
        
        scr[j] = score(data_i[j], labels_i[j], th, th0)/n
        
    return scr.mean()

   
    