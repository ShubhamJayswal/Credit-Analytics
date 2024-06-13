# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:48:29 2024

@author: Shubham Jayswal
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_excel('CreditCardDatasetNew.xlsx')

# Calculate 10% of the data
sample_size = int(len(data) * 0.1)

# Perform stratified sampling to get 10% of the rows
sampled_data, _ = train_test_split(data, train_size=sample_size, stratify=data['is_fraud'], random_state=42)

# Save the sampled dataset to a new Excel file
sampled_data.to_excel('sampled_dataset_10_percent.xlsx', index=False)