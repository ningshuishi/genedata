#!/usr/bin/env python
# -*- coding:utf-8 -*- 
#Author: sunweiwei
#Time : 2018/12/21 9:24
'''
#根据relief算法，计算基因的重要性排序
'''
import pdb
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.NaN)

#read data and split into two class
train_data = pd.read_csv('./with24_data_norm.csv',index_col=0)
train_data=train_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.1.1'])

class1_data = train_data.loc[train_data['alive_year'] <= 3]
class2_data = train_data.loc[train_data['alive_year'] > 3]
class1_data = class1_data.drop(['alive_year'], axis=1)    # (74, 40540)
class2_data = class2_data.drop(['alive_year'], axis=1)    # (57, 40540)

#filter feature extraction
class1_row_name = class1_data.index.tolist()  # get the row names of class1_data
delta = np.zeros(class1_data.shape[1])       # to save the importance of each gene , shape :(18491,)

for list_i in class1_row_name:  # iterate each sample
    data_i = class1_data.loc[list_i,:].values   # current sample
    hit_samples = class1_data.drop([list_i]).values    # samples of the same class as the current sample
    miss_samples = class2_data.values                  # samples of the different class as the current sample
    near_hit = np.fabs(data_i - hit_samples).min(axis=0)  # near hit for each feature , shape :(41870,)
    near_miss = np.fabs(data_i - miss_samples).min(axis=0)  # near miss for each feature , shape :(41870,)
    near_sum = near_miss - near_hit
    delta += near_sum  # accumulate the result of each sample

gene_importance = pd.DataFrame(delta,index= class1_data.columns.values,columns=['importance'])
sorted_gene_importance = gene_importance.sort_values(by='importance',ascending=False)  # sort the importance of genes
sorted_gene_importance.to_csv('./gene_importance_relief_with24.csv')  # save gene importance files

