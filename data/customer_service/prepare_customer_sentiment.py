#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:49:09 2025

@author: keremyasar
"""

import os
import pandas as pd
import tiktoken
import numpy as np
from sklearn.model_selection import train_test_split

# Path to CSV file  
input_file_path = '/Users/keremyasar/Desktop/DI725-Assignment-1/data/customer_service/train.csv'  

# Load the CSV file
df = pd.read_csv(input_file_path)

# Display the first few rows to understand the data structure
print("Data preview:")
print(df.head())
print(f"Total records: {len(df)}")

# Combine sentiment and conversation for encoding
# This approach keeps the relationship between sentiment and conversation intact
df['combined'] = df['customer_sentiment'].astype(str) + " | " + df['conversation'].astype(str)

# Split into training and validation sets (90% training, 10% validation)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Function to process and save data
def process_and_save(data_df, output_filename):
    # Join all text into a single string for encoding
    combined_text = "\n".join(data_df['combined'].tolist())
    
    # Encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(combined_text)
    
    print(f"{output_filename} has {len(ids):,} tokens")
    
    # Convert to numpy array and save
    ids_array = np.array(ids, dtype=np.uint16)
    output_path = os.path.join(os.path.dirname(__file__), output_filename + '.bin')
    ids_array.tofile(output_path)
    print(f"Saved to {output_path}")
    
    # Also save the data as CSV for reference or different usage
    csv_output_path = os.path.join(os.path.dirname(__file__), output_filename + '.csv')
    data_df.to_csv(csv_output_path, index=False)
    print(f"Also saved as CSV to {csv_output_path}")

# Process and save training data
process_and_save(train_df, 'train')

# Process and save validation data
process_and_save(val_df, 'val')

print("Data processing and splitting complete!")