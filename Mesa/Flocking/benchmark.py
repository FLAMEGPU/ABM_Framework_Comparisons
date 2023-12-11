# Use Python 3

# Only collect the number of wolves and sheeps per step.

import timeit
import gc
import statistics
import random

REPETITIONS = 3
SEED = 12

random.seed(SEED)
a = []
for i in range(0, REPETITIONS):
    setup=f"""
gc.enable()
import os, sys
sys.path.insert(0, os.path.abspath("."))

from model import BoidFlockers

def runthemodel(flock):
    for i in range(0, 100):
      flock.step()


flock = BoidFlockers(
  population=80000,
  width=400,
  height=400,
  seed={random.randint(0, 999999999)}
)
"""
    tt = timeit.Timer('runthemodel(flock)', setup=setup)
    a.append(tt.timeit(1))
print("Mesa Flocking times (ms):", list(map(lambda x: x * 1e3, a)))
print("Mesa Flocking (mean ms):", statistics.mean(a)*1e3)

