# common
import os
import numpy as np
import pandas as pd
from glob import glob
import torch
import random
y1 = []
list1=np.array([random.randint(0,10) for i in range(10)])
list2=np.array([random.randint(0,10) for i in range(20)])

y1.append(list1)
y1.append(list2)

print(y1)
# print(y1)
