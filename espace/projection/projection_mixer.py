'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: projection mixer file.
'''

import json
import torch
import sys
import os

from espace.generic.model_modifier import check_module_name_in_target

def load_pt_dict(file_name):
  print("loading from {}".format(file_name))
  pt_dict=torch.load(file_name,map_location=torch.device('cuda:0'))
  print("loaded from {}".format(file_name))
  return pt_dict

def save_pt_dict(pt_dict,file_name):
  print("saving the dict to {}".format(file_name))
  torch.save(pt_dict, file_name)
  print("saved the dict to {}".format(file_name))

if __name__=='__main__':
  proj_dir=sys.argv[1]
  destination_file = os.path.join(proj_dir,sys.argv[2])
  destination_dict = {}
  num_layers=0
  configuration_file = sys.argv[3]
  with open(configuration_file,'r') as f:
    configuration_dict = json.load(f)
    # the json should look like { "proj": [ "target1", "target2",...]}

  for source_file_name, targets in configuration_dict.items():
    source_dict = load_pt_dict(os.path.join(proj_dir,source_file_name))
    for module_name in source_dict:
      for target in targets:
        if check_module_name_in_target(module_name,target):
          print("Grabbing {} from {}".format(module_name,source_file_name))
          destination_dict[module_name]=source_dict[module_name]
          num_layers+=1
          print("Number of layers grabbed so far: {}".format(num_layers))
    print("\n\n")

  save_pt_dict(destination_dict,destination_file)
