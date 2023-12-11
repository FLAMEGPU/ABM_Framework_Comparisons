import csv, os, re
import matplotlib.pyplot as plt
import numpy as np

regex = re.compile('testwrap[0-9]*.csv')

x = []
y = []
fx = []
fy = []
color = []
for root, dirs, files in os.walk(os.getcwd()):
  for file in files:
    if regex.match(file):
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            next(reader, None)  # skip the headers
            for row in reader:
                x.append(row[0])
                y.append(row[1])
                fx.append(row[2])
                fy.append(row[3])
                color.append(((row[2] + 1.0)/2, (row[3] + 1.0)/2, 0.0))
        
plt.scatter(np.array(x), np.array(y), s=2, c=color, marker=",", linewidths=0.1)
plt.show()