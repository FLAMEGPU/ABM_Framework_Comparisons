# Use Python 3

# Only collect the number of wolves and sheeps per step.

import timeit
import gc

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
SAMPLES=10
a = (sum(tt.repeat(SAMPLES, 1)))/SAMPLES
print("Mesa Flocking (ms):", a*1e3)

