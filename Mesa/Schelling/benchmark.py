# Use Python 3

import timeit
import gc
import statistics

setup = f"""
gc.enable()
import os, sys
sys.path.insert(0, os.path.abspath("."))

from model import SchellingModel

import random
random.seed(2)

def runthemodel(schelling):
    for i in range(0, 100):
      schelling.step()

schelling = SchellingModel(
  height=500,
  width=500,
  density=0.8
)
"""

tt = timeit.Timer('runthemodel(schelling)', setup=setup)
SAMPLES=3
a = tt.repeat(SAMPLES, 1)
print("Mesa schelling times (ms):", list(map(lambda x: x * 1e3, a)))
print("Mesa schelling (mean ms):", statistics.mean(a)*1e3)


