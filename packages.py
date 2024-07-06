import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#import operator
#from functools import reduce
#from functools import partial

import os, platform
import copy
from math import ceil, floor
from timeit import default_timer
from datetime import datetime