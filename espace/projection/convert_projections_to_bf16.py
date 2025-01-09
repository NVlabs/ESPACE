'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: projection conversion file.
'''

import json
import torch
import sys
import os

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
  source_file = os.path.join(proj_dir,sys.argv[2])
  destination_file = os.path.join(proj_dir,sys.argv[3])
  destination_dict = {}
  source_dict = load_pt_dict(source_file)
  for module_name,module_dict in source_dict.items():
    destination_module_dict={}
    print("looking into {}".format(module_name))
    for key,val in module_dict.items():
      print("converting {}".format(key))
      destination_module_dict[key]=val.to(torch.bfloat16)
      print("\n")
    destination_dict[module_name]=destination_module_dict

  save_pt_dict(destination_dict,destination_file)
