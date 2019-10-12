# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:00:08 2019

@author: Lyd
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xlwt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('./with24_data_norm.csv',index_col=0)
importance=pd.read_csv('./gene_importance_relief_with24.csv',index_col=0)
#data_to_add=pd.read_csv('E:/genedata/WorkinVacation/with_clinical_dropzero_nolessthan2years_01A_norm_2-4.csv',index_col=0)
#df = pd.DataFrame(columns = ["mae", "rmse", "acc","c_index"]) 
result=[]
for i in range(len(data)):
    data.loc[i,'alive_year'] = 0 if data.loc[i, 'alive_year'] <=2 else 1
#train_data1=data[0:33]
#train_data2=data[66:]
test_data=data[99:]
train_data=data[0:99]
#train_data=train_data1.append(train_data2)
for t in range(200):
    importance_gene_name=importance.iloc[0:t+1,0:1]
    n_gene_name=importance_gene_name.index.tolist()
    #X=data[n_gene_name]
    #Y=data['alive_year']
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2)
    X_train=train_data[n_gene_name]
    y_train=train_data['alive_year']
    X_test=test_data[n_gene_name]
    y_test=test_data['alive_year']
    
    clf=RandomForestClassifier()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    
    predict_prob_y = clf.predict_proba(X_test)[:,1]
    fpr,tpr,threshold=roc_curve(y_test,predict_prob_y)
    test_auc=auc(fpr,tpr)    
    #test_auc = metrics.roc_auc_score(y_test,predict_prob_y)
    all=[acc,test_auc,(t+1)]
    result.append(all)
    print t,acc,test_auc
 
output = open('./kfold/rf4.xls','w')
output.write('acc\tauc\tfeature_num\n')
for i in range(len(result)):
	for j in range(len(result[i])):
		output.write(str(result[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
		output.write('\t')   #相当于Tab一下，换一个单元格
	output.write('\n')       #写完一行立马换行
output.close()
