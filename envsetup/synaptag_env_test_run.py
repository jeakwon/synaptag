from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import requests
from io import BytesIO
import timm
import time

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

print('successful')