"""
Originally forked from Andrej Karpathy's minGPT.

CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None
    # TODO: [part g]
    ### YOUR CODE HERE ###
    positions = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1) # shape (T, 1)
    dims = torch.arange(dim//2, dtype=torch.float32).unsqueeze(0) # shape (1, dim/2)
    thetas = 10000.0**(-2.0*(dims)/dim) # (1, dim/2)
    angles = positions * thetas # Dimension broadcasting -> (T, dim2)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rope_cache = torch.stack((cos, sin), dim=-1)
    
    ### END YOUR CODE ###
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """
    Apply the RoPE to the input tensor x.
    x is key or query vector as rope is applied to k, q, not word embeddings
    x has shape (Batch, num_heads, T, dim of each head)
    """
    # TODO: [part g]
    # You might find the following functions useful to convert
    # between real and complex numbers:

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    rotated_x = None
    ### YOUR CODE HERE ###
    B, num_head, T, dim_per_head = x.shape
    # Split x into real and imaginary part, where all odd
    # original x has shape (B, num_heads, T, dim_per_head)
    real_x = x[:,:,:,0::2] # all even indexed x's dimensions
    imaginary_x = x[:,:,:,1::2] # all odd indexed x's dimensions
    pre_complex_x = torch.stack((real_x, imaginary_x), dim=-1) # (B, num_heads, T, dim_per_head/2, 2)

    # fetch the precomupted sin and cos
    rope_cache = rope_cache[:T, :, :] # truncate the rope_cache to the length of the input sequence
    cos = rope_cache[:, :, 0]
    sin = rope_cache[:, :, 1]
    cossin = torch.stack((cos, sin), dim=-1).unsqueeze(dim=0).unsqueeze(dim=0) # (T, dim/2, 2) -> (1, 1, T, dim/2, 2)

    # Compute the RoPE as element-wise multiplication
    x_complex = torch.view_as_complex(pre_complex_x) # (B, num_heads, T, dim_per_head/2, 2) -> (B, num_heads, T, dim_per_head/2)
    cossin_complex = torch.view_as_complex(cossin) # (1, 1, T, dim/2, 2) -> (1, 1, T, dim/2)
    rotated_x = x_complex * cossin_complex # will broadcase dimension and perform ELEMENTWISE MULTIPLICATION
                                           # (B, num_heads, T, dim_per_head/2)

    rotated_x_real = torch.view_as_real(rotated_x) # (B, num_heads, T, dim_per_head/2) -> (B, num_heads, T, dim_per_head/2, 2)
    rotated_x = rotated_x_real.flatten(-2) # flatten the last two dimensions (B, num_heads, T, dim_per_head/2, 2) -> (B, num_heads, T, dim_per_head/2*2)
                                                # the dimensions will be flattened in turns, i.e. real0, imag0, real1, imag1, ...
    
    ### END YOUR CODE ###
    return rotated_x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            # TODO: [part g] Precompute the cos and sin values for RoPE and
            # store them in rope_cache.
            # Hint: The maximum sequence length is given by config.block_size.
            rope_cache = None
            ### YOUR CODE HERE ###
            rope_cache = precompute_rotary_emb(
                dim=config.n_embd//config.n_head, # the embedding dimensions is equally distributed to all heads
                max_positions=config.block_size
            ) 
            
            ### END YOUR CODE ###

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope:
            # TODO: [part g] Apply RoPE to the query and key.
            ### YOUR CODE HERE ###
            q = apply_rotary_emb(q, self.rope_cache)
            k = apply_rotary_emb(k, self.rope_cache)
            ### END YOUR CODE ###

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    """
    Modifications over the self-attention layer to handle two inputs and perform
    cross-attention between them.
    This follows the implementation of the self attention module with
    auto-regressive masking on (key).
    Manipulation of batch-size to allow for different batch size between the 
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        # keys of x1
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)
        
        # query with x2
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2) # (B, nh, Tq, hs)
        
        # values from x1
        v = self.value(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        B = max(Bk, Bq)
        
        att = att.masked_fill(self.mask[:,:,:Tq,:Tk] == 0, -1e10) 
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, Cq) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
