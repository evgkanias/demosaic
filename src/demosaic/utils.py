"""
General utility functions and static variables.
"""
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2026, Lund Vision Group, Lund University"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0"
__maintainer__ = "Evripidis Gkanias"

import yaml
import os

__ROOT_DIR__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(__ROOT_DIR__, "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

UINT16_MAX = 65520.0
LDR_MAX = 65535.0
UINT8_MAX = 255.0
