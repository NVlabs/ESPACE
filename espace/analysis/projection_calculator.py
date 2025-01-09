'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: projection calculator file.
'''
import torch
import cupy
import numpy as np

import torch.nn as nn
import os
import logging
logger=logging.getLogger(__name__)

from espace.generic.arguments import espace_args 

import time


#here I am assuming that the full name of the file is given, including rank and extension
def load_statistics():
  logger.info("loading statistics from {}".format(espace_args.statistic_save_fname))
  statistics_dictionary=torch.load(espace_args.statistic_save_fname,map_location=torch.device('cuda:0'))
  logger.info("-loaded statistics dictionary")
  return statistics_dictionary

def save_projections(projections_dictionary):
  logger.info("saving projections to {}".format(espace_args.projection_save_fname))
  torch.save(projections_dictionary,espace_args.projection_save_fname)
  logger.info("-Saved projections dictionary")

def check_eig_decomposition(original_matrix,eigen_values,eigen_vectors):
  reconstruction_matrix = cupy.dot(cupy.dot(eigen_vectors,cupy.diag(eigen_values)),cupy.transpose(eigen_vectors))
  delta_matrix = cupy.absolute(reconstruction_matrix-original_matrix)
  return cupy.mean(delta_matrix),cupy.amax(delta_matrix)

def check_orthonormal(matrix):
  vvt = cupy.dot(matrix,cupy.transpose(matrix))
  delta_matrix=cupy.absolute(vvt-cupy.eye(cupy.shape(matrix)[0]))
  return cupy.mean(delta_matrix),cupy.amax(delta_matrix)

def calculate_projections():
  statistics_dictionary=load_statistics()
  projections_dictionary={}
  times=[]
  for module_name, stat_matrix in statistics_dictionary.items():
    projections_dictionary[module_name]={}
    logger.info("Calculating projection for layer {}".format(module_name))
    if stat_matrix.dtype is not torch.float:
      storage_dtype=stat_matrix.dtype
      capsule=torch.utils.dlpack.to_dlpack(stat_matrix.to(torch.float))
    else:
      storage_dtype=torch.float
      capsule=torch.utils.dlpack.to_dlpack(stat_matrix)
    if espace_args.projection_method=="eigen":
      logger.info("-We will apply eigenvalue decomposition to the stat matrix of size {}".format(stat_matrix.size()))
      start_time=time.time()
      stat_cupy=cupy.from_dlpack(capsule)
      logger.info("-Data has now been moved from torch to cupy")
      eigen_values, eigen_vectors = cupy.linalg.eigh(stat_cupy)
      logger.info("-Eigen values have been computed")
      elapsed_time=time.time()-start_time
      if storage_dtype is torch.float:
        projections_dictionary[module_name]['eigen_vectors']=torch.utils.dlpack.from_dlpack(eigen_vectors.toDlpack())
      else:
        projections_dictionary[module_name]['eigen_vectors']=torch.utils.dlpack.from_dlpack(eigen_vectors.toDlpack()).to(storage_dtype)
      projections_dictionary[module_name]['eigen_values']=torch.utils.dlpack.from_dlpack(eigen_values.toDlpack())
      logger.info("--Size of eigen values is {}  and size of eigen vectors is {}".format(projections_dictionary[module_name]['eigen_values'].size(),projections_dictionary[module_name]['eigen_vectors'].size()))
      logger.info("---Check that this is an eigen decomposition, absolute avg and max errors of reconstruction are {}".format(check_eig_decomposition(stat_cupy,eigen_values,eigen_vectors)))
      logger.info("---Check that eigen vectors are orthonormal, absolute avg and max errors of VVT with Eye are {}".format(check_orthonormal(eigen_vectors)))
      logger.info("\n")
      #logger.info("---Sneak peak at the eigen values: {}".format(eigen_values))
    times.append(elapsed_time)
    logger.info("---Done with one projection in {} seconds".format(elapsed_time))

  times=np.asarray(times)
  logger.info("Time to finish everything is {} seconds, and average time per projection is {} seconds".format(np.sum(times),np.mean(times)))
  save_projections(projections_dictionary)

