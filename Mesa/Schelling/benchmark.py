# Use Python 3

import timeit
import gc

setup = f"""
gc.enable()
import os, sys
sys.path.insert(0, os.path.abspath("."))

from model import SchellingModel

import random
random.seed(2)

def runthemodel(schelling):
    for i in range(0, 10):
      schelling.step()


schelling = SchellingModel(
  height=500,
  width=500,
  density=0.8
)
"""

tt = timeit.Timer('runthemodel(schelling)', setup=setup)
SAMPLES=10
a = (sum(tt.repeat(SAMPLES, 1)))/SAMPLES
print("Mesa Schelling (ms):", a*1e3)

