# Use Python 3

import timeit
import gc
import statistics
import random

REPETITIONS = 3
SEED = 12

random.seed(SEED)
a = []
for i in range(0, REPETITIONS):
    setup = f"""
gc.enable()
import os, sys
sys.path.insert(0, os.path.abspath("."))

from model import SchellingModel

def runthemodel(schelling):
    for i in range(0, 100):
      schelling.step()

schelling = SchellingModel(
  height=500,
  width=500,
  density=0.8,
  seed={random.randint(0, 999999999)}
)
"""
    tt = timeit.Timer('runthemodel(schelling)', setup=setup)
    a.append(tt.timeit(1))
print("Mesa schelling times (ms):", list(map(lambda x: x * 1e3, a)))
print("Mesa schelling (mean ms):", statistics.mean(a)*1e3)


