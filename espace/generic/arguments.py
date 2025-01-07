'''
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

ESPACE Toolkit: arguments file.
'''
from argparse import Namespace
import sys
import logging
logger = logging.getLogger(__name__)

global espace_args
espace_args = Namespace()

def parse_espace_arguments(args: Namespace):
  """ Parsing of espace args
  Args:
    args:  a namespace object inherited from the top level application
  """
  
  for arg in vars(args):
    if arg[:7] == "espace_":
      setattr(espace_args, arg[7:], getattr(args,arg))
    else:
      logging.info(f"Warning: an argument with incorrect prefix was passed to the espace toolkit: {arg}")
  
  #set defaults if they don't exist
  if not hasattr(espace_args, "statistic") or espace_args.statistic is None : espace_args.statistic="autocorrelation"
  if not hasattr(espace_args, "normalize_during_tracking") or espace_args.normalize_during_tracking is None : espace_args.normalize_during_tracking=None
  if not hasattr(espace_args, "normalize_during_projection") or espace_args.normalize_during_projection is None : espace_args.normalize_during_projection=None
  if not hasattr(espace_args, "mode") or espace_args.mode is None : espace_args.mode="estimation"
  if not hasattr(espace_args, "target_modules") or espace_args.target_modules is None : espace_args.target_modules=None
  #if not hasattr(espace_args, "result_directory") or espcace_args.result_directory is None : espace_args.result_directory="/result" #not really needed
  if not hasattr(espace_args, "logging_fname") or espace_args.logging_fname is None : espace_args.logging_fname="/result/espace_logging"
  if not hasattr(espace_args, "statistic_save_fname") or espace_args.statistic_save_fname is None : espace_args.statistic_save_fname=None
  if not hasattr(espace_args, "projection_save_fname") or espace_args.projection_save_fname is None : espace_args.projection_save_fname=None
  if not hasattr(espace_args, "projection_method") or espace_args.projection_method is None : espace_args.projection_method="eigen"
  if not hasattr(espace_args, "reconstruction_analysis_save_fname") or espace_args.reconstruction_analysis_save_fname is None : espace_args.reconstruction_analysis_save_fname="/metrics/reconstruction_metrics"
  if not hasattr(espace_args, "precision") or espace_args.precision is None : espace_args.precision="fp32"
  if not hasattr(espace_args, "merge_to_TP") or espace_args.merge_to_TP is None : espace_args.merge_to_TP=None
  if not hasattr(espace_args, "secondary_mode") or espace_args.secondary_mode is None : espace_args.secondary_mode=None
  if not hasattr(espace_args, "secondary_target_modules") or espace_args.secondary_target_modules is None : espace_args.secondary_target_modules=None
  if not hasattr(espace_args, "outermost_target_module") or espace_args.outermost_target_module is None : espace_args.outermost_target_module=None


def print_espace_arguments():
  # Print all the arguments into te logger
  logger.info("--------------------------------------------------------------------------")
  logger.info("ESPACE Arguments")
  logger.info("--------------------------------------------------------------------------")
  
  for arg in vars(espace_args):
    logger.info("{}:\t{}".format(arg,getattr(espace_args,arg)))

  logger.info("--------------------------------------------------------------------------")

def parse_console_args():
  logger.info("Parsing Console Arguments")
  keys=sys.argv[1::2]
  values=sys.argv[2::2]
  args_dict = {k[2:]:v for k,v in zip(keys,values)}
  return Namespace(**args_dict)

