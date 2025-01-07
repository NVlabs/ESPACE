'''
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

ESPACE Toolkit: helper file.
'''
import torch
import sys
import pickle

def load_dict(file_prefix):
  fr=file_prefix+".pth"
  print("loading from {}".format(fr))
  pt_dict=torch.load(fr)
  print("loaded from {}".format(fr))
  return pt_dict

def to_numpy(pt_dict):
  for key in pt_dict:
    if isinstance(pt_dict[key],dict):
      print("recursing through {}".format(key))
      to_numpy(pt_dict[key])
    else:
      pt_dict[key]=pt_dict[key].numpy(force=True)
      print("converted {}".format(key))

def save_pkl(np_dict,file_prefix):
  fw=file_prefix+".pkl"
  print("saving to {}".format(fw))
  with open(fw,'wb') as f:
    pickle.dump(np_dict,f)
  print("saved to {}".format(fw))


if __name__=='__main__':
  file_prefixes=sys.argv[1:]
  for file_prefix in file_prefixes:
    current_dict=load_dict(file_prefix)
    to_numpy(current_dict)
    print(current_dict)
    save_pkl(current_dict,file_prefix)
