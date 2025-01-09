'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: TP merge file.
'''
import torch
import logging
import os
import json
import glob
logger = logging.getLogger(__name__)
from espace.generic.arguments import espace_args
from espace.generic.name_checking_utils import check_module_name_in_target

def load_fully_merged_projections(device):
  merged_projections={}
  available_projection_files = glob.glob(espace_args.projection_save_fname + "_rank_*.pth")
  num_projection_files = len(available_projection_files)

  #first load rank 0
  fr = espace_args.projection_save_fname + "_rank_0.pth"
  rank_0_projections = torch.load(fr,map_location=device)
  logger.info("-Loaded rank 0 projections")
  for module_name, projection_matrix in rank_0_projections.items():
    merged_projections[module_name]={}
    merged_projections[module_name]['eigen_vectors']=projection_matrix['eigen_vectors']
    merged_projections[module_name]['eigen_values']=projection_matrix['eigen_values']
    logger.info("--Initialized projection for module {}".format(module_name))
  del rank_0_projections
  logger.info("-Done with initialization from rank 0\n")

  for rank in range(1,num_projection_files):
    fr = espace_args.projection_save_fname + "_rank_{}.pth".format(rank)
    current_rank_projections = torch.load(fr,map_location=device)
    logger.info("-Adding matrices from rank {}".format(rank))
    for module_name in merged_projections:
      if check_module_name_in_target(module_name,"attention__AND__dense__OR__4h_to_h__OR__linear_proj__OR__linear_fc2"):
        logger.info("--Stacking matrices at module {}".format(module_name))
        merged_projections[module_name]['eigen_vectors']=torch.cat((merged_projections[module_name]['eigen_vectors'],current_rank_projections[module_name]['eigen_vectors']),0)
        merged_projections[module_name]['eigen_values']=torch.cat((merged_projections[module_name]['eigen_values'],current_rank_projections[module_name]['eigen_values']),0)
        if rank == num_projection_files-1:
          current_shape=merged_projections[module_name]['eigen_vectors'].shape
          merged_projections[module_name]['eigen_vectors']=torch.reshape(merged_projections[module_name]['eigen_vectors'],(num_projection_files,current_shape[0]//num_projection_files,current_shape[1])) #it's now (TP,K,K) - second K becomes truncated to L
          current_shape=merged_projections[module_name]['eigen_values'].shape
          merged_projections[module_name]['eigen_values']=torch.reshape(merged_projections[module_name]['eigen_values'],(num_projection_files,current_shape[0]//num_projection_files)) #(TP,K)
#      else:
#        logger.info("--Averaging matrices at module {}".format(module_name))
#        merged_projections[module_name]['eigen_vectors']=merged_projections[module_name]['eigen_vectors']+current_rank_projections[module_name]['eigen_vectors']
#        merged_projections[module_name]['eigen_values']=merged_projections[module_name]['eigen_values']+current_rank_projections[module_name]['eigen_values']
#        if rank == num_projection_files-1:
#          merged_projections[module_name]['eigen_vectors']=merged_projections[module_name]['eigen_vectors']/num_projection_files
#          merged_projections[module_name]['eigen_values']=merged_projections[module_name]['eigen_values']/num_projection_files
    logger.info("-Added matrices from rank {}\n".format(rank))
    del current_rank_projections

  return merged_projections
