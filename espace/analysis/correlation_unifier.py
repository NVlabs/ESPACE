'''
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

ESPACE Toolkit: correlation unifier file.
'''
import torch
import glob
import os
import logging
logger=logging.getLogger(__name__)

from espace.generic.arguments import espace_args 
from espace.generic.name_checking_utils import check_module_name_in_target 


#here I am assuming that the full name of the file is given, including rank and extension
def load_statistics(fname):
  logger.info("loading statistics from {}".format(fname))
  statistics_dictionary=torch.load(fname,map_location=torch.device('cuda:0'))
  logger.info("-loaded statistics dictionary")
  return statistics_dictionary

def save_statistics(stat_dict,fname):
  logger.info("saving Stats to {}".format(fname))
  torch.save(stat_dict,fname)
  logger.info("-Saved projections dictionary")


def unify_correlations():
  accumulator_dictionary={}
  number_of_ranks = len(glob.glob(espace_args.statistic_save_fname + "_rank_*.pth"))
  logger.info("Found {} rank file".format(number_of_ranks))

  logger.info("Initializing the accumulators from rank 0")
  rank_0_stats_dict=load_statistics(espace_args.statistic_save_fname + "_rank_0.pth")
  for module_name, stat_matrix in rank_0_stats_dict.items():
    if check_module_name_in_target(module_name,espace_args.target_modules):
      accumulator_dictionary[module_name] = stat_matrix
      logger.info("-Inititalized {}".format(module_name))
  del rank_0_stats_dict

  for rank in range(1,number_of_ranks):
    logger.info("Adding stats from rank {}".format(rank))
    current_rank_stat_dict=load_statistics(espace_args.statistic_save_fname + "_rank_{}.pth".format(rank))
    for module_name in accumulator_dictionary:
      accumulator_dictionary[module_name] = accumulator_dictionary[module_name] + current_rank_stat_dict[module_name]
      logger.info("-Added stat for module: {}".format(module_name))
    del current_rank_stat_dict

  logger.info("Unifying stat matrices")
  for rank in range(number_of_ranks):
    logger.info("Unifying stats at rank {}".format(rank))
    current_rank_stat_dict=load_statistics(espace_args.statistic_save_fname + "_rank_{}.pth".format(rank))
    for module_name in accumulator_dictionary:
      current_rank_stat_dict[module_name]=accumulator_dictionary[module_name]/number_of_ranks
      logger.info("-Unified stats for module: {}".format(module_name))
    save_fname = espace_args.statistic_save_fname + "_unified_rank_{}.pth".format(rank)
    save_statistics(current_rank_stat_dict,save_fname)
    del current_rank_stat_dict

