"""
Created on Sat Jan 11 15:11:24 2020

@author: hamza
"""

#Uppper Confidence Bound

#importing the libraries
import pandas as pd
import numpy as np
import math
from functools import reduce

    
def all_rounds(dataset, d, n, default_confidence, default_average):        
    '''
    Parameters
    ----------

    dataset:
        row -> total rounds(n)
        col -> total column (d)

    d: Total arms 'd':
        i.e; Total no. of ads, etc. 

    n: Total rounds 'n':
        i.e; Total no. of visitors who visited web page.

    default_confidence: (By defalut: 0.5)

    default_average: (By default: 0.5)

    Return
    ------

    n_Of_visited: ndarray of visited ads
        i.e; Who visited this page

    reward: nd.array that 
        i.e; Reward from each visitor for ads

    choosen_ads:
        i.e;  Reward for each ads
    '''

    number_times = [0] * d
    sum_reward = [0] * d
    winner_index = 0
    selected_ad = []
    for i in range(0,n):
        number_times[winner_index] += 1
        sum_reward[winner_index] += dataset.values[i, winner_index]
        average_reward = []
        confidences = []
        upper_bound = []
        selected_ad.append(winner_index)
        for i in range(0,d):
            #Error handling in case number_times == 0
            try: 
                average = (sum_reward[i]/number_times[i])
            except:
                average = default_average
            try:
                confidence = math.sqrt(1.5*math.log(i+1)/number_times[i])
            except:
                confidence = default_confidence
                
            average_reward.append(average)
            confidences.append(confidence)
            upper_bound.append(average_reward[i] + confidences[i])
        
        winner = max(upper_bound)
        winner_index = upper_bound.index(winner)
        
    return (number_times, sum_reward, selected_ad)





