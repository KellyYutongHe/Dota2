
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()


# In[2]:

sentimental_table = pd.read_csv('sentimental.csv')
senti_word_list = list(sentimental_table)[1:]


# In[3]:

for index in range(len(sentimental_table["often"])):
    if sentimental_table["often"][index] == "Very often":
        sentimental_table.set_value(index, "often", 1)
    if sentimental_table["often"][index] == "Sometimes":
        sentimental_table.set_value(index, "often", 0.75)
    if sentimental_table["often"][index] == "Seldom, but I know Dota 2":
        sentimental_table.set_value(index, "often", 0.5)
sentimental_table


# In[13]:

weights = list(sentimental_table["often"])
sentimental_scores = []
for word in senti_word_list:
    weighted_values = []
    old_values = list(sentimental_table[word])
    for index in range(len(weights)):
        if old_values[index] != "I don't know":
            old_value = float(old_values[index])
#             print(word)
#             print(old_value)
            weight = weights[index]
#             print(weight)
            new_value = old_value*weight
#             print(new_value)
            weighted_values.append(new_value)
    average = sum(weighted_values)/float(len(weighted_values))
    sentimental_scores.append(average)
sentimental_dict = dict(zip(senti_word_list, sentimental_scores))
sentimental_dict

