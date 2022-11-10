# Use Python 3

# Only collect the number of wolves and sheeps per step.

import timeit
import gc

setup = f"""
gc.enable()
import os, sys
sys.path.insert(0, os.path.abspath("."))

from agents import Sheep, Wolf, GrassPatch
from model import WolfSheep

def runthemodel(wolfsheep):
    for i in range(0, 500):
      wolfsheep.step()

wolfsheep = WolfSheep()
"""

tt = timeit.Timer('runthemodel(wolfsheep)', setup=setup)
SAMPLES=10
a = (sum(tt.repeat(SAMPLES, 1)))/SAMPLES
print("Mesa WolfSheep (ms):", a*1e3)
