# Use Python 3

# Only collect the number of wolves and sheeps per step.

import timeit
import gc
import statistics

setup = f"""
gc.enable()
import os, sys
sys.path.insert(0, os.path.abspath("."))

from model import BoidFlockers

def runthemodel(flock):
    for i in range(0, 100):
      flock.step()


flock = BoidFlockers(
  population=30000,
  width=100,
  height=100
)
"""

tt = timeit.Timer('runthemodel(flock)', setup=setup)
SAMPLES=5
a = tt.repeat(SAMPLES, 1)
print("Mesa Flocking times (ms):", list(map(lambda x: x * 1e3, a)))
print("Mesa Flocking (mean ms):", statistics.mean(a)*1e3)

