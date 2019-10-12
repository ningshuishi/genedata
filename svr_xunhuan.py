# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:30:19 2018

@author: Lyd
"""

import pandas as pd
from sklearn import svm
from sklearn.svm import SVR
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
data=pd.read_csv('./with24_data_norm.csv',index_col=0)
importance=pd.read_csv('./gene_importance_relief_with24.csv',index_col=0)
#data_to_add=pd.read_csv('E:/genedata/WorkinVacation/with_clinical_dropzero_nolessthan2years_01A_norm_2-4.csv',index_col=0)
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
    
    svr_lin = SVR(kernel='linear', C=0.5) 
    svr_poly = SVR(kernel='poly', C=30, degree=1) 
    #svr_poly = SVR(kernel='rbf', C=104, gamma=0.1)
    y_lin = svr_lin.fit(X_train, y_train).predict(X_test) 
    y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
    Y_lin=list(y_lin)
    Y_poly=list(y_poly)
    true = list(y_test)

    pd_true=pd.DataFrame(true)
    pd_Y_lin=pd.DataFrame(Y_lin)
    pd_Y_poly=pd.DataFrame(Y_poly)
    pd_true.columns = ["a"]
    pd_Y_lin.columns = ["a"]
    pd_Y_poly.columns = ["a"]
    pd_true = pd_true.reindex(columns=list('ab'), fill_value=1)
    pd_Y_lin = pd_Y_lin.reindex(columns=list('ab'), fill_value=1)
    pd_Y_poly = pd_Y_poly.reindex(columns=list('ab'), fill_value=1)
    for i in range(len(true)):
        if pd_true.iloc[i,0]<3:
            pd_true.iloc[i,1]=0
        else:
            pd_true.iloc[i,1]=1
    for i in range(len(true)):
        if pd_Y_lin.iloc[i,0]<3:
            pd_Y_lin.iloc[i,1]=0
        else:
            pd_Y_lin.iloc[i,1]=1
        
    for i in range(len(true)):
        if pd_Y_poly.iloc[i,0]<3:
            pd_Y_poly.iloc[i,1]=0
        else:
            pd_Y_poly.iloc[i,1]=1
            
    m=0
    n=0            
    for i in range(len(true)):
        if pd_Y_lin.iloc[i,1]==pd_true.iloc[i,1]:
            m+=1
        else:
            m=m            
    for i in range(len(true)):
        if pd_Y_poly.iloc[i,1]==pd_true.iloc[i,1]:
            n+=1
        else:
            n=n        
    linacc=float(m)/float((i+1))
    polyacc=float(n)/float((i+1))       
    linMAE=metrics.mean_absolute_error(true, Y_lin)
    linRMSE=np.sqrt(metrics.mean_squared_error(true, Y_lin))
    polyMAE=metrics.mean_absolute_error(true, Y_poly)
    polyRMSE=np.sqrt(metrics.mean_squared_error(true, Y_poly))
    
    #c-index
    count_lin=0
    sum_lin=0
    for i in range(len(Y_lin)):
        for j in range(i+1,len(true)):
            if i==j:
                print i
            else:
                sum_lin+=1
                if Y_lin[i]<=Y_lin[j] and true[i]<=true[j]:
                    count_lin+=1
                elif Y_lin[i]>Y_lin[j] and true[i]>true[j]:
                    count_lin+=1
#                else:
#                    print count_lin
    c_index_lin=float(count_lin)/sum_lin
    
     #c-index
    count_poly=0
    sum_poly=0
    for i in range(len(y_poly)):
        for j in range(i+1,len(true)):
            if i==j:
                print i
            else:
                sum_poly+=1
                if y_poly[i]<=y_poly[j] and true[i]<=true[j]:
                    count_poly+=1
                elif y_poly[i]>y_poly[j] and true[i]>true[j]:
                    count_poly+=1
#                else:
#                    print count_poly
    c_index_poly=float(count_poly)/sum_poly
    
    
    
    all=[linMAE,linRMSE,linacc,c_index_lin,polyMAE,polyRMSE,polyacc,c_index_poly,(t+1)]      
    result.append(all)
    print t,linacc,polyacc
    
'''
output = open('./kfold/svr4.xls','w')
output.write('linMAE\tlinRMSE\tlinacc\tc_index_lin\tpolyMAE\tpolyRMSE\tpolyacc\tc_index_poly\tfeature_num\n')
for i in range(len(result)):
	for j in range(len(result[i])):
		output.write(str(result[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
		output.write('\t')   #相当于Tab一下，换一个单元格
	output.write('\n')       #写完一行立马换行
output.close()            
'''
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            