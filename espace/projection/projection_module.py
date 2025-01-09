'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: projection module file.
'''

import torch
from torch import linalg as LA
import torch.nn as nn
import os
import logging
logger=logging.getLogger(__name__)
import numpy as np

from espace.generic.arguments import espace_args 

@torch.jit.script
def TP_merged_forward(projection_matrix,input):
  matricized_input = torch.flatten(input, start_dim=0, end_dim=-2) #now of shape (M,K*TP)
  previous_tp_size = projection_matrix.shape[0]
  reshaped_input = torch.reshape(matricized_input,(previous_tp_size,matricized_input.shape[0],matricized_input.shape[1]//previous_tp_size)) #(TP,M,K)
  #proj matrix has shape(TP,K,L)
  pojection_stack = torch.bmm(reshaped_input,projection_matrix) #of size (TP,M,L)
  matricized_projections = torch.reshape(pojection_stack,(pojection_stack.shape[1],pojection_stack.shape[0]*pojection_stack.shape[2]))
  #tensorized_projections = torch.reshape(matricized_projections,new_shape)
  return matricized_projections

class ProjectionModule(nn.Module):
  def __init__(self,
      module_name,
      projection_matrix,
      scores,
      num_components
      ):
    super(ProjectionModule,self).__init__()

    self.module_name = module_name
    self.initialize_configurators(num_components)
    self.initialize_buffers(projection_matrix,scores)
  
  def initialize_configurators(self,num_components):
    self.num_components=num_components
    self.projection_method = espace_args.projection_method
    if espace_args.precision == "bf16":
      self.precision = torch.bfloat16
    elif espace_args.precision == "fp32":
      self.precision = torch.float
    elif espace_args.precision == "fp16":
      self.precision = torch.float16
    
  def initialize_buffers(self,loaded_projection_matrix,scores):
    #no matter whether TP mergin or not, pytorch's sort function sorts the last dimension, that's what we want
    if "eigen" in self.projection_method:
      if self.projection_method == "absolute_eigen":
        sorted_eigen_value_indices = torch.argsort(torch.abs(scores),descending=True)
      elif self.projection_method == "eigen":
        sorted_eigen_value_indices = torch.argsort(scores,descending=True)

    if len(loaded_projection_matrix.shape)==3:
      #TP merging mode: projection matrix should be (TP,K,K) and scores (TP,K)
      if len(scores.shape)!=2:
        logger.info("!!! There's something wrong with the shapes of eigen vectors and eigenvalues !!!")
      self.TP_merging=True
      truncated_indices = sorted_eigen_value_indices[:,:self.num_components] # these guys now have shape (TP,L)
      truncated_proj_matrix = torch.gather(loaded_projection_matrix,2,truncated_indices[:,None,:].repeat(1,loaded_projection_matrix.shape[1],1)) 
      #this guy gather from (TP,K,K) using (TP,None,L) indices -> (TP,K,L)
      self.register_buffer('projection_matrix',truncated_proj_matrix.to(self.precision)) #has shape (TP,K,L)
      del truncated_proj_matrix
      del truncated_indices
      logger.info("Here is the shape of the projection matrix: {}".format(self.projection_matrix.shape))
    
    else:
      #normal mode
      self.TP_merging=False
      truncated_indices = sorted_eigen_value_indices[:self.num_components]
      self.register_buffer('projection_matrix',loaded_projection_matrix[:,truncated_indices].to(self.precision)) # no need to worry about transposition for weights because megatron passes weights tranposed
    del loaded_projection_matrix

  def forward(self,input):
    if self.TP_merging:
      original_input_shape = input.shape
      new_shape = tuple(list(original_input_shape[:-1])+[self.projection_matrix.shape[0]*self.projection_matrix.shape[2]])
      matricized_projections = TP_merged_forward(self.projection_matrix,input)
      return torch.reshape(matricized_projections,new_shape)
    else:
      return torch.matmul(input,self.projection_matrix)

class CommonProjectionModule(nn.Module):
  def __init__(self):
    super(CommonProjectionModule,self).__init__()
    self.enabled = False

  def enable_common_projection(self,
      projection_matrix,
      scores,
      num_components
      ):
    self.enabled = True
    #self.module_name = module_name
    self.initialize_configurators(num_components)
    self.initialize_buffers(projection_matrix,scores)
  
  def initialize_configurators(self,num_components):
    self.num_components=num_components
    self.projection_method = espace_args.projection_method
    if espace_args.precision == "bf16":
      self.precision = torch.bfloat16
    elif espace_args.precision == "fp32":
      self.precision = torch.float
    elif espace_args.precision == "fp16":
      self.precision = torch.float16
    
  def initialize_buffers(self,loaded_projection_matrix,scores):
    #no matter whether TP mergin or not, pytorch's sort function sorts the last dimension, that's what we want
    if "eigen" in self.projection_method:
      if self.projection_method == "absolute_eigen":
        sorted_eigen_value_indices = torch.argsort(torch.abs(scores),descending=True)
      elif self.projection_method == "eigen":
        sorted_eigen_value_indices = torch.argsort(scores,descending=True)

    if len(loaded_projection_matrix.shape)==3:
      #TP merging mode: projection matrix should be (TP,K,K) and scores (TP,K)
      if len(scores.shape)!=2:
        logger.info("!!! There's something wrong with the shapes of eigen vectors and eigenvalues !!!")
      self.TP_merging=True
      truncated_indices = sorted_eigen_value_indices[:,:self.num_components] # these guys now have shape (TP,L)
      truncated_proj_matrix = torch.gather(loaded_projection_matrix,2,truncated_indices[:,None,:].repeat(1,loaded_projection_matrix.shape[1],1)) 
      #this guy gather from (TP,K,K) using (TP,None,L) indices -> (TP,K,L)
      self.register_buffer('projection_matrix',truncated_proj_matrix.to(self.precision)) #has shape (TP,K,L)
      del truncated_proj_matrix
      del truncated_indices
      logger.info("Here is the shape of the projection matrix: {}".format(self.projection_matrix.shape))
    
    else:
      #normal mode
      self.TP_merging=False
      truncated_indices = sorted_eigen_value_indices[:self.num_components]
      self.register_buffer('projection_matrix',loaded_projection_matrix[:,truncated_indices].to(self.precision)) # no need to worry about transposition for weights because megatron passes weights tranposed
    del loaded_projection_matrix

  def forward(self,act,weight):
    projected_act = torch.matmul(act,self.projection_matrix)
    projected_weight = torch.matmul(weight,self.projection_matrix)
    return projected_act, projected_weight


class PreComputedWeightProjection(nn.Module):
  def __init__(self, weight):
    super(PreComputedWeightProjection,self).__init__()
    self.original_weight = weight
    self.precompute_enabled = False

  def initialize_configurators(self,num_components):
    self.num_components=num_components
    self.projection_method = espace_args.projection_method
    if espace_args.precision == "bf16":
      self.precision = torch.bfloat16
    elif espace_args.precision == "fp32":
      self.precision = torch.float
    self.original_weight.to(self.precision)

  def get_projection_matrix(self,loaded_projection_matrix,scores):
    if "eigen" in self.projection_method:
      if self.projection_method == "absolute_eigen":
        sorted_eigen_value_indices = torch.argsort(torch.abs(scores),descending=True)
      elif self.projection_method == "eigen":
        sorted_eigen_value_indices = torch.argsort(scores,descending=True)
    if len(loaded_projection_matrix.shape)==3:
      #TP merging mode: projection matrix should be (TP,K,K) and scores (TP,K)
      if len(scores.shape)!=2:
        logger.info("!!! There's something wrong with the shapes of eigen vectors and eigenvalues !!!")
      self.TP_merging=True
      truncated_indices = sorted_eigen_value_indices[:,:self.num_components] # these guys now have shape (TP,L)
      truncated_proj_matrix = torch.gather(loaded_projection_matrix,2,truncated_indices[:,None,:].repeat(1,loaded_projection_matrix.shape[1],1)) 
      #this guy gather from (TP,K,K) using (TP,None,L) indices -> (TP,K,L)
      projection_matrix=truncated_proj_matrix.to(self.precision) #has shape (TP,K,L)
      del truncated_proj_matrix
      del truncated_indices
    
    else:
      #normal mode
      self.TP_merging=False
      truncated_indices = sorted_eigen_value_indices[:self.num_components]
      projection_matrix=loaded_projection_matrix[:,truncated_indices].to(self.precision) # no need to worry about transposition for weights because megatron passes weights tranposed
    del loaded_projection_matrix
    logger.info("Here is the shape of the projection matrix: {}".format(projection_matrix.shape))
    return projection_matrix

  def enable_precomputation(self,loaded_projection_matrix,scores,num_components):
    self.precompute_enabled = True
    self.initialize_configurators(num_components)
    projection_matrix = self.get_projection_matrix(loaded_projection_matrix,scores)

    if self.TP_merging:
      original_input_shape = self.original_weight.shape
      new_shape = tuple(list(original_input_shape[:-1])+[projection_matrix.shape[0]*projection_matrix.shape[2]])
      matricized_projections = TP_merged_forward(projection_matrix,self.original_weight)
      self.register_buffer('precomputed_weight_projection', torch.reshape(matricized_projections,new_shape))
    else:
      self.register_buffer('precomputed_weight_projection',torch.matmul(self.original_weight,projection_matrix))
    del self.original_weight
    del projection_matrix
    logger.info("Here is the shape of the precomputed weight times projection {}".format(self.precomputed_weight_projection.shape))

  def forward(self, input):#pass non at application
    return self.precomputed_weight_projection
    
    

