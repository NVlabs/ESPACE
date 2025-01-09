'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: analysis util file.
'''
import os

from espace.analysis.projection_calculator import calculate_projections
from espace.analysis.correlation_unifier import unify_correlations
from espace.generic.arguments import espace_args, parse_espace_arguments, print_espace_arguments, parse_console_args
from espace.generic.logger_handling import configure_loggers_file_handlers

if __name__=='__main__':
  args=parse_console_args()
  parse_espace_arguments(args)
  configure_loggers_file_handlers()
  print_espace_arguments()

  if espace_args.mode=="projection_preparation":
    print("We are going to compute projections based on estimated statistics")
    calculate_projections()

  if espace_args.mode=="correlation_unification":
    print("We are going to unify correlation matrices across ranks")
    unify_correlations()
