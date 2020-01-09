#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 22:42:32 2020

@author: hamza
"""
"""
Case: A product purchased atleast 3 times a day if we found association 
       of that product X e.g; 3 product would be bought in a row if X product 
       would bought.
       
For Support
-----------
purchase of product X of a day = 3
purchase of product X of a week = 3x7 = 21
transaction per week of product X = 7500
    ∴ Support = 3x7/7500 = 0.0028 ≅ 0.003
            
For Confidence
--------------
Possibility of association of product X with product Y is called confedence. 
Bydefault min. confidence is 80%(0.8) but we divideit by;
    ∴ 0.8/4 = 0.2(20%)

Lift
----
The ratio of Confidence and Support is called lift.
    ∴ Best min. lift = 3
    
Note: This also depends upon business problem to choose Lift values.
 
"""
# Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

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

# Step 3: Training Apriori on the dataset
# Step 4: Sortted already with apriori module
from apyori import apriori
rules = apriori(transactions,       # 2D list that contain items of each customer of customers
                min_support=0.003,  # item/Transaction (I) /Total no. of items/Transactions
                min_confedence=0.2, # 20% posiility of association with products 
                min_lift=3,         # confidence/support = lift (higher is better) 
                min_length=2)       # Min. 2 products for association rule 

# Steps 5: Visualization of result
rules = list(rules)
listRules = [list(rules[i][0]) for i in range(0,len(rules))]
