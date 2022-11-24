# Use Python 3

import timeit
import gc
import statistics

setup = f"""
gc.enable()
import os, sys
sys.path.insert(0, os.path.abspath("."))

from model import ForestFire

import random
random.seed(2)

def runthemodel(fire):
    for i in range(0, 100):
      fire.step()


fire = ForestFire()
"""

tt = timeit.Timer('runthemodel(fire)', setup=setup)
SAMPLES=5
a = tt.repeat(SAMPLES, 1)
print("Mesa Flocking times (ms):", list(map(lambda x: x * 1e3, a)))
print("Mesa Flocking (mean ms):", statistics.mean(a)*1e3)

