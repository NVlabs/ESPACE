'''
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

 NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 property and proprietary rights in and to this material, related
 documentation and any modifications thereto. Any use, reproduction,
 disclosure or distribution of this material and related documentation
 is subject to the NVIDIA license agreement.

ESPACE Toolkit: logging file.
'''
import logging
import os
logger = logging.getLogger(__name__)

from espace.generic.arguments import espace_args

import time

FORMAT = "%(asctime)s [%(levelname)-7s][%(funcName)26s] %(message)s"
formatter = logging.Formatter(FORMAT, "%m-%d %H:%M")

def configure_loggers_file_handlers():
  # prepare file handlers, each rank will write a log file
  fh = logging.FileHandler(os.path.join(espace_args.logging_fname+"_rank_{}.log".format(os.environ.get("RANK",'0'))),mode='w')
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)

  for name, item in logging.root.manager.loggerDict.items():
    if type(item) is logging.Logger and "espace" in name:
      item.addHandler(fh)
      item.propagate = False
      item.setLevel(logging.DEBUG)
  time.sleep(1)
