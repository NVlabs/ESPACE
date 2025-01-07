'''
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

ESPACE Toolkit: statistics estimator file.
'''

import torch
from torch import linalg as LA
import torch.nn as nn
import os
import logging
logger=logging.getLogger(__name__)

from espace.generic.arguments import espace_args 

def compute_batched_autocorrelation_matrix(
    input: torch.Tensor,
    normalize: str = None,
) -> torch.Tensor:
  """ Computing the batched autocorrelation matrix given one input tensor. 
  The vectors are always assumed to be on the last dimension (K), and all leading dimensions are stacked into one (M). The autocorrelation matrix is
  computed as C = 1/M X^T X.

  Args:
    input: input tensor, i.e., X
    normalize: optional normalization technique applied to the vectors before computing the matrix.
      normalizations supported: 
        None: do not do any normalization
        "L2_norm": normalize each vector by its L2 norm
  """
  
  X = torch.flatten(input.to(torch.bfloat16), start_dim=0, end_dim=-2) #cast to bf16 is to reduce memory consumption while staying accurate
  M = X.shape[0] # although not needed we also have K = X.shape[1]

  if not (normalize is None):
    if normalize == "L2_norm":
      X_norms = LA.vector_norm(X,ord=2,dim=1)
      X = torch.nan_to_num(X/X_norms[:,None])
      # nan_to_num is used in case a vector has zero norm
    else:
      raise ValueError(f"Unknown Normalization Technique: {normalize}")

  return torch.matmul(X.t(),X)/M


class TensorStatisticsEstimator(nn.Module):
  def __init__(self,
      module_name
      ):
    super(TensorStatisticsEstimator,self).__init__()

    self.module_name = module_name

    self.initialize_buffers()
    self.initialize_trackers()
    self.initialize_configurators()


  def initialize_buffers(self):
    self.register_buffer('accumulated_autocorrelation_matrix', None)

  def initialize_trackers(self):
    self.elapsed_batches=0

  def initialize_configurators(self):
    self.statistic = espace_args.statistic
    self.normalize = espace_args.normalize_during_tracking
    if self.statistic in ["grad_autocorrelation"]:
      self.gradient_stat_accumulator = self.gradient_stat_fn()

  def gradient_stat_fn(self):

    class stat_fn(torch.autograd.Function):

      @staticmethod
      def forward(ctx, input):
        if self.statistic == "grad_autocorrelation":
          input_autocorr = compute_batched_autocorrelation_matrix(input,self.normalize)
          ctx.save_for_backward(input_autocorr)
          
        return input

      @staticmethod
      def backward(ctx, grad_output):
        if self.statistic == "grad_autocorrelation":
          grad_autocorr = compute_batched_autocorrelation_matrix(grad_output,self.normalize)
          input_autocorr, = ctx.saved_tensors
          new_autocorrelation_matrix = torch.matmul(input_autocorr,grad_autocorr)+torch.matmul(grad_autocorr,input_autocorr)

        if self.accumulated_autocorrelation_matrix is None:
          self.accumulated_autocorrelation_matrix = new_autocorrelation_matrix
        else:
          self.accumulated_autocorrelation_matrix = self.accumulated_autocorrelation_matrix + new_autocorrelation_matrix
        return grad_output

    return stat_fn().apply


  def forward(self, input):
    self.elapsed_batches = self.elapsed_batches + 1
    if "autocorrelation" in self.statistic:

      if self.statistic in ["grad_autocorrelation"]:
        input.requires_grad_(True)
        buffered_input = self.gradient_stat_accumulator(input)
        return buffered_input

      else:
        batched_autocorrelation_matrix = compute_batched_autocorrelation_matrix(input,self.normalize)
        if self.accumulated_autocorrelation_matrix is None:
          self.accumulated_autocorrelation_matrix = batched_autocorrelation_matrix
        else:
          self.accumulated_autocorrelation_matrix = self.accumulated_autocorrelation_matrix + batched_autocorrelation_matrix

    return input

def save_estimated_statistics(model):
  statistics_dictionary={}
  for name, module in model.named_modules():

    if espace_args.statistic in ["autocorrelation", "grad_autocorrelation"]:
      if type(module) == TensorStatisticsEstimator:
        logger.info("Saving {} matrix estimated at layer {}.".format(espace_args.statistic,name))
        statistics_dictionary[name] = module.accumulated_autocorrelation_matrix/module.elapsed_batches
        logger.info("Saved a matrix of shape {}, and type {}".format(statistics_dictionary[name].size(),statistics_dictionary[name].dtype))

    elif espace_args.statistic == "scaled_autocorrelation":
      found_input_autocorr=False
      found_weight_autocorr=False
      input_autocorr=None
      weight_autocorr=None
      for child_name, child_module in module.named_children():

        if "input_identity" in child_name and type(child_module) == TensorStatisticsEstimator:
          found_input_autocorr=True
          input_autocorr = child_module.accumulated_autocorrelation_matrix/child_module.elapsed_batches
          logger.info("Grabbed an estimated autocorrelation from the input child of {}".format(name))

        elif "weight_identity" in child_name and type(child_module) == TensorStatisticsEstimator:
          found_weight_autocorr=True
          weight_autocorr = child_module.accumulated_autocorrelation_matrix/child_module.elapsed_batches
          logger.info("Grabbed an estimated autocorrelation from the weight child of {}".format(name))

      if found_input_autocorr and found_weight_autocorr:
        statistics_dictionary[name+".input_identity"] = torch.matmul(weight_autocorr,input_autocorr)+torch.matmul(input_autocorr,weight_autocorr)
        logger.info("Save CxCw+CwCx, the matrix has size {}, and type {}".format(statistics_dictionary[name+".input_identity"].size(),statistics_dictionary[name+".input_identity"].dtype))

    else:
        logger.info("Could not save statistic at layer {}. No implementation for {} tracking.".format(name,espace_args.statistic))

  fsuffix="_rank_{}.pth".format(os.environ["RANK"])
  fw = espace_args.statistic_save_fname + fsuffix
  logger.info("saving all statistics to {}".format(fw))
  torch.save(statistics_dictionary,fw)

