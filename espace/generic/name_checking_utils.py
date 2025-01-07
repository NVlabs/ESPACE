'''
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

ESPACE Toolkit: helper file.
'''
import torch
import logging
import os
import json
import glob
logger = logging.getLogger(__name__)
from espace.generic.arguments import espace_args



def check_module_name_in_target(module_name,target_substrings):
  # look for __NOT__
  yes_no_targets = target_substrings.split('__NOT__')
  if len(yes_no_targets)==1:
    return check_module_name_matches_target_expression(module_name, target_substrings)
  elif len(yes_no_targets)==2:
    yes_targets = yes_no_targets[0]
    no_targets = yes_no_targets[1]
    if check_module_name_matches_target_expression(module_name, no_targets):
      return False
    else:
      return check_module_name_matches_target_expression(module_name, yes_targets)
  else:
    logger.info("Please do not use __NOT__ more than once")
  
def check_module_name_matches_target_expression(module_name, target_substrings):
  # Check if 'module_name' is in the targeted names
  # 'target_substrings' are parsed using __AND__ and __OR__, the former being checked first
  # substrings are needed, so no need to specify the full name of the layer
  #
  # example, search for activation_identity__OR__weight_identity for module model.module.layer.0.weight_identity
  
  for disjuncted_target_substring in target_substrings.split('__OR__'):
    target_match = True
    for conjuncted_target_substring in disjuncted_target_substring.split('__AND__'):
      if not(conjuncted_target_substring in module_name): 
        target_match = False
        break
    if target_match == True:
      return True

  return False
