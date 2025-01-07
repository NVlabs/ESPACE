'''
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

ESPACE Toolkit: model modifier file.
'''
import torch
import logging
import os
import json
import glob
logger = logging.getLogger(__name__)

from espace.analysis.statistics_estimator import TensorStatisticsEstimator
from espace.analysis.reconstruction_analyzer import ReconstructionAnalyzer
from espace.projection.projection_module import ProjectionModule
from espace.generic.arguments import espace_args, parse_espace_arguments, print_espace_arguments
from espace.generic.logger_handling import configure_loggers_file_handlers
from espace.generic.tp_merge_utils import load_fully_merged_projections
from espace.generic.name_checking_utils import check_module_name_in_target 

def get_layer(model, name):
  # Get the layer names 'name' in the 'model'. Follows MPQT style.
  # example: try to get module.layer.0.weight_identity
  # loop over [module, layer, 0, weight_identity]
  # get model.module, then module.layer, then layer.0, then finally 0.weight_identity
  layer = model
  for attr in name.split("."):
    layer = getattr(layer, attr)
  return layer

def set_layer(model, name, layer):
  # Set the layer named 'name' to 'layer' in 'model'. Follows MPQT style.
  
  attrs, name = name.rsplit(".",1)
  model = get_layer(model, attrs)
  # this will make 'model' point to the parent of layer 'name'
  # in the example above, model will become model.module.layer.0 and we will set its "weight_identity" to "layer"
  
  setattr(model,name,layer)


def extract_projection_configuration(name):
  # with this, we may provide the arg as --espace_target_modules "{'query':'128','dense':'256'}"
  target_modules_str_dict = espace_args.target_modules
  apply_espace = False
  num_components = None
  target_modules_dict = json.loads(target_modules_str_dict)
  for target_substring, num_component_str in target_modules_dict.items():
    if check_module_name_in_target(name,target_substring):
      apply_espace=True
      num_components=int(num_component_str)
      return apply_espace,num_components
  return apply_espace,num_components

def load_projections(device):
  if espace_args.merge_to_TP == 1:
    logger.info("We will merge projections from multiple ranks to 1")
    return load_fully_merged_projections(device)
  elif espace_args.merge_to_TP is not None:
    logger.info("We have still not implemented merging projections to a TP other than 1")
  current_rank = int(os.environ["RANK"])
  available_projection_files = glob.glob(espace_args.projection_save_fname + "_rank_*.pth")
  num_projection_files = len(available_projection_files)
  rank_to_load = current_rank % num_projection_files
  logger.info("There are {} projection files found, we are on rank {}, so we will load from rank {} in the spirit of modulo".format(
    num_projection_files,current_rank,rank_to_load))
  fsuffix="_rank_{}.pth".format(rank_to_load)
  fr = espace_args.projection_save_fname + fsuffix
  logger.info("-loading projection matrices from {}".format(fr))
  projections_dictionary=torch.load(fr,map_location=device)
  logger.info("--loaded projections dictionary")
  return projections_dictionary



def modify_model(model, args):
  # Modify 'model' so as to replace specific layers by desired ones
  # args are top level arguments provided
  
  device = next(model.parameters()).device
  
  parse_espace_arguments(args)
  configure_loggers_file_handlers()
  print_espace_arguments()

  #if espace_args.mode in ["reconstruction_analysis","projection_forward","projection_precomputed","projection_common","distillation_adaptation"]:
  if espace_args.projection_save_fname is not None:
    projections_dictionary = load_projections(device)
  else:
    projections_dictionary = None

  for name, module in model.named_modules():

    if espace_args.mode in ["projection_forward", "projection_precomputed","projection_common"]:
      apply_espace,num_components = extract_projection_configuration(name)
    elif "distillation" in espace_args.mode: # and espace_args.projection_method != "load_distiller":
       apply_espace,num_components = extract_projection_configuration(name)
    else:
      apply_espace = check_module_name_in_target(name,espace_args.target_modules)

    if apply_espace:
      new_layer=None
      logger.info(f"ESPACE will be applied to module: {name}")
      old_layer = get_layer(model,name)

      if espace_args.mode == "estimation":
        new_layer = TensorStatisticsEstimator(name)
        logger.info(f"{name} will be set as a Tensor Statistics Estimator")

      elif espace_args.mode == "reconstruction_analysis":
        if espace_args.projection_method == "eigen":
          eigen_vectors = projections_dictionary[name]['eigen_vectors']
          eigen_values = projections_dictionary[name]['eigen_values']
          new_layer = ReconstructionAnalyzer(name,eigen_vectors,eigen_values)
          logger.info(f"{name} will be set as a Reconstruction Analyzer")

      elif espace_args.mode == "projection_common":
        if "eigen" in espace_args.projection_method:
          if name.endswith("common_identity"):
            dict_key = name[:-15]+"input_identity"
            eigen_vectors = projections_dictionary[dict_key]['eigen_vectors']
            eigen_values = projections_dictionary[dict_key]['eigen_values']
            module.enable_common_projection(eigen_vectors,eigen_values,num_components)
            logger.info(f"{name} is enabling Common Projection with {num_components} components")

      elif espace_args.mode == "projection_forward":
        if "eigen" in espace_args.projection_method:
          if name.endswith("weight_identity"):
            dict_key = name[:-15]+"input_identity"
          elif name.endswith("input_identity"):
            dict_key=name
          else:
            continue
          eigen_vectors = projections_dictionary[dict_key]['eigen_vectors']
          eigen_values = projections_dictionary[dict_key]['eigen_values']
          new_layer = ProjectionModule(name,eigen_vectors,eigen_values,num_components)
          logger.info(f"{name} will be set as a Projection Module with {num_components} components")

      elif espace_args.mode == "projection_precomputed":
        if "eigen" in espace_args.projection_method:
          if name.endswith("precompute_identity"):
            dict_key = name[:-19]+"input_identity"
            eigen_vectors = projections_dictionary[dict_key]['eigen_vectors']
            eigen_values = projections_dictionary[dict_key]['eigen_values']
            module.enable_precomputation(eigen_vectors,eigen_values,num_components)
            logger.info(f"{name} enabled pre-computation of weight times projection with {num_components} components")
          elif name.endswith("input_identity"):
            dict_key=name
            eigen_vectors = projections_dictionary[dict_key]['eigen_vectors']
            eigen_values = projections_dictionary[dict_key]['eigen_values']
            new_layer = ProjectionModule(name,eigen_vectors,eigen_values,num_components)
            logger.info(f"{name} will be set as a Projection Module with {num_components} components")

      else:
        logger.info(f"ESPACE mode {espace_args.espace_mode} is not implemented")

      if new_layer is None:
        logger.info(f"{name} will remain of type {type(module)}")
      else:
        set_layer(model,name,new_layer)
        del old_layer

  if projections_dictionary is not None:
    del projections_dictionary
    #torch.cuda.empty_cache()
  model.to(device)
  logger.info("Model modification done. Everything is moved to {}, which is where the model initially was.".format(device))





