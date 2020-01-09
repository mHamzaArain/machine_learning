#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:31:12 2020

@author: hamza
"""

# Step 1: import libraries
import numpy as np
import pandas as pd

# Step 2: Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',names=np.arange(1,21))

# Step 2(a): Preprocesing data to fit model
"""
transactions = []
for sublist in dataset.values.tolist():    
    print(sublist)    #  ['mineral water', nan, nan,......, nan]
    clean_sublist = [item for item in sublist if item is not np.nan]  # ['mineral water']
    transactions.append(clean_sublist) # remove 'nan' values # https://github.com/rasbt/mlxtend/issues/433
"""
transactions = [
            [product for product in products if product is not np.nan] 
            for products in dataset.values.tolist()]

# One hot
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()                               # transaction input X dataset 
te_arry = te.fit(transactions).transform(transactions)  # format dataset suitable for ML APIs via fit() method.
                                                        #   Learns the unique labels in the dataset (True\Fales),
                                                        #   and via the transform() method
  
df_x = pd.DataFrame(te_arry,             # row=containing set of bool for specific product 
                    columns=te.columns_) # col=all products are arranged in alphabatical order

# Step 4: Train model using Apiori algorithm 
# ref = https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets = apriori(df_x,               # values are either 0/1 or True/False.
                  min_support=0.005,  # a set of transactions containing(I)/transactions
                  use_colnames=True)  # Allowed col

df_rules = association_rules(df_sets,           # values are either 0/1 or True/False.
                             metric='support',  # (bydefault cofidence) but is support formula 
                             min_threshold= 0.005, # 0.5%
                             support_only=True) # It is support for eclat rule

# if you use only "support", it called "ECLAT"















