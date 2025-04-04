#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:47:52 2025

@author: keremyasar
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('/Users/keremyasar/Desktop/DI725-Assignment-1/data/customer_service/train.csv')
df_test = pd.read_csv('/Users/keremyasar/Desktop/DI725-Assignment-1/data/customer_service/test.csv')


print(df_test.columns)
value_counts_test = df_test['customer_sentiment'].value_counts()
value_counts_train = df_train['customer_sentiment'].value_counts()


value_counts_test.plot(kind='bar')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Value counts of customer sentiment')

# Display the plot
plt.tight_layout()
plt.show()

value_counts_train.plot(kind='bar')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Value counts of customer sentiment')

# Display the plot
plt.tight_layout()
plt.show()