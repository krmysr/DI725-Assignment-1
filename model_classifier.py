#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:15:41 2025

@author: keremyasar
"""

"""
Modified version of the GPT model that outputs 3 classes instead of next-token prediction.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import the original GPT model
from model import GPT, GPTConfig

class GPTClassifier(GPT):
    """GPT model with a classification head instead of a language modeling head."""
    
    def __init__(self, config):
        # Initialize the parent GPT class
        super().__init__(config)
        
        # Remove the original language modeling head
        del self.lm_head
        
        # Create a new classification head for 3 classes
        self.classifier_head = nn.Linear(config.n_embd, 3, bias=True)
        
        # Initialize the weights for the classifier head
        torch.nn.init.normal_(self.classifier_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.classifier_head.bias)
        
        # Add a pooling strategy 
        self.pooling = 'last'
        
        # Recalculate and report the number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        Args:
            idx (torch.LongTensor): Input token indices, shape [batch_size, sequence_length]
            targets (torch.LongTensor, optional): Target class labels (0, 1, or 2), shape [batch_size]
        
        Returns:
            tuple: (logits, loss) where logits are the class probabilities and loss is the classification loss
                  if targets are provided, otherwise loss is None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward the GPT model itself (same as original)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Apply pooling to get a single vector per sequence
        if self.pooling == 'last':
            # Use the last token's representation
            pooled = x[:, -1, :]  # shape: [b, n_embd]
        elif self.pooling == 'mean':
            # Average across the sequence dimension
            pooled = x.mean(dim=1)  # shape: [b, n_embd]
        elif self.pooling == 'max':
            # Max pooling across the sequence dimension
            pooled = x.max(dim=1)[0]  # shape: [b, n_embd]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Pass through classification head
        logits = self.classifier_head(pooled)  # shape: [b, 3]
        
        # If targets are provided, calculate classification loss
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
            
        return logits, loss
    
    def set_pooling(self, pooling_type):
        """
        Set the pooling strategy for the classifier.
        
        Args:
            pooling_type (str): One of 'last', 'mean', or 'max'
        """
        assert pooling_type in ['last', 'mean', 'max'], f"Pooling must be one of ['last', 'mean', 'max'], got {pooling_type}"
        self.pooling = pooling_type
        print(f"Pooling strategy set to: {pooling_type}")