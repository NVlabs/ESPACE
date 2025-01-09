'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: reconstruciton analyzer file.
'''

import torch
from torch import linalg as LA
import torch.nn as nn
import os
import logging
logger=logging.getLogger(__name__)
import numpy as np

from espace.generic.arguments import espace_args 

class ReconstructionAnalyzer(nn.Module):
  def __init__(self,
      module_name,
      projection_matrix,
      scores
      ):
    super(ReconstructionAnalyzer,self).__init__()

    self.module_name = module_name

    self.initialize_configurators()
    self.initialize_buffers(projection_matrix,scores)
    self.initialize_trackers()

  def initialize_buffers(self,loaded_projection_matrix, scores=None):
    if self.projection_method == "eigen":
      eigen_values=scores
      self.num_components=list(eigen_values.size())[0]
      #principal_components_sorted_indices=torch.argsort(eigen_values,descending=True)
      self.register_buffer('projection_matrix',torch.flip(loaded_projection_matrix,[1])) #cupy eigh returned eigenvalues in increaing order
    self.num_candidate_truncations=int(np.floor(np.log2(self.num_components)))
    self.register_buffer('accumulated_mses',torch.zeros(self.num_candidate_truncations).float())
    self.register_buffer('accumulated_nmses',torch.zeros(self.num_candidate_truncations).float())
    self.register_buffer('accumulated_squared_mses',torch.zeros(self.num_candidate_truncations).float())
    self.register_buffer('accumulated_squared_nmses',torch.zeros(self.num_candidate_truncations).float())
    logger.info("Reconstruction MSEs will be analyzed for {} using {} candidate components out of {}".format(self.module_name,self.num_candidate_truncations,self.num_components))

  def initialize_trackers(self):
    self.elapsed_batches=0

  def initialize_configurators(self):
    self.projection_method = espace_args.projection_method

  def forward(self, input):
    self.elapsed_batches+=1
    for k in range(self.num_candidate_truncations):
      flattened_input=torch.flatten(input,start_dim=0,end_dim=-2)
      candidate_projection_matrix = self.projection_matrix[:,:int(np.exp2(k+1))]
      reconstruction=torch.matmul(torch.matmul(flattened_input,candidate_projection_matrix),torch.t(candidate_projection_matrix))
      delta = reconstruction-flattened_input
      reconstruction_mse = torch.mean(torch.square(LA.vector_norm(delta,ord=2,dim=1)))
      input_norm = LA.vector_norm(flattened_input,ord=2,dim=1)
      normalized_delta = torch.nan_to_num(delta/input_norm[:,None])
      reconstruction_nmse = torch.mean(torch.square(LA.vector_norm(normalized_delta,ord=2,dim=1)))
      self.accumulated_mses[k]+=reconstruction_mse
      self.accumulated_nmses[k]+=reconstruction_nmse
      self.accumulated_squared_mses[k]+=torch.square(reconstruction_mse)
      self.accumulated_squared_nmses[k]+=torch.square(reconstruction_nmse)
    return input


def save_reconstruction_analysis(model):
  reconstruction_metrics_dictionary={}
  for name, module in model.named_modules():
    if type(module) == ReconstructionAnalyzer:
      logger.info(f"Saving reconstruction metrics at layer {name}.")
      reconstruction_metrics_dictionary[name] = {}
      reconstruction_metrics_dictionary[name]['avg_mses']=module.accumulated_mses/module.elapsed_batches
      reconstruction_metrics_dictionary[name]['avg_nmses']=module.accumulated_nmses/module.elapsed_batches
      reconstruction_metrics_dictionary[name]['avg_square_mses']=module.accumulated_squared_mses/module.elapsed_batches
      reconstruction_metrics_dictionary[name]['avg_square_nmses']=module.accumulated_squared_nmses/module.elapsed_batches
      for metric_name, metric in reconstruction_metrics_dictionary[name].items():
        logger.info("--{} are saved, they are equal to {}".format(metric_name,metric))
      logger.info("-Saved!")

  fsuffix="_rank_{}.pth".format(os.environ["RANK"])
  fw = espace_args.reconstruction_analysis_save_fname + fsuffix
  logger.info("saving all metrics to {}".format(fw))
  torch.save(reconstruction_metrics_dictionary,fw)

