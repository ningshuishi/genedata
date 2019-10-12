# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:25:57 2018

@author: Lyd
"""

import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xlwt
from sklearn.linear_model import RidgeCV


data=pd.read_csv('./with24_data_norm.csv',index_col=0)
importance=pd.read_csv('./gene_importance_relief_with24.csv',index_col=0)
result=[]

#train_data1=data[0:66]
#train_data2=data[99:]
test_data=data[99:]
train_data=data[0:99]
#train_data=train_data1.append(train_data2)

for t in range(200):
    importance_gene_name=importance.iloc[0:t+1,0:1]
    n_gene_name=importance_gene_name.index.tolist()
    #X=data[n_gene_name]
    #Y=data['alive_year']
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    X_train=train_data[n_gene_name]
    y_train=train_data['alive_year']
    X_test=test_data[n_gene_name]
    y_test=test_data['alive_year']
    
    clf = Ridge(alpha=2)
    y_pred=clf.fit(X_train,y_train).predict(X_test)
    true = list(y_test)
    pred=list(y_pred)
    pd_true=pd.DataFrame(true)
    pd_pred=pd.DataFrame(pred)
    pd_true.columns = ["a"]
    pd_pred.columns = ["a"]
    pd_true = pd_true.reindex(columns=list('ab'), fill_value=1)
    pd_pred = pd_pred.reindex(columns=list('ab'), fill_value=1)
    for i in range(len(true)):
         if pd_true.iloc[i,0]<3:
             pd_true.iloc[i,1]=0
         else:
             pd_true.iloc[i,1]=1

    for i in range(len(true)):
        if pd_pred.iloc[i,0]<3:
            pd_pred.iloc[i,1]=0
        else:
            pd_pred.iloc[i,1]=1
    m=0
    for i in range(len(true)):
        if pd_pred.iloc[i,1]==pd_true.iloc[i,1]:
            m+=1
        else:
            m=m
    acc=float(m)/float((i+1)) 
    mae=metrics.mean_absolute_error(true, pred)
    rmse=np.sqrt(metrics.mean_squared_error(true, pred))
    np.mean(true)
    
    #c-index
    count=0
    sum=0
    for i in range(len(pred)):
        for j in range(i+1,len(true)):
            if i==j:
                print i
            else:
                sum+=1
                if pred[i]<=pred[j] and true[i]<=true[j]:
                    count+=1
                elif pred[i]>pred[j] and true[i]>true[j]:
                    count+=1
#                else:
#                    print count
    c_index=float(count)/sum
    
    
    all=[mae,rmse,acc,c_index,(t+1)]
    result.append(all)
    print t,mae,acc,c_index
    

output = open('./kfold/Ridge_regression4.xls','w')
output.write('mae\tmase\tacc\tc_index\tfeature_num\n')
for i in range(len(result)):
	for j in range(len(result[i])):
		output.write(str(result[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
		output.write('\t')   #相当于Tab一下，换一个单元格
	output.write('\n')       #写完一行立马换行
output.close()





