from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
  
x = []
y = []
  

df = pd.read_csv('results/framevariations.csv')
df = df.drop('Unnamed: 0', axis=1)
columns = list(df.columns)
dicts = {}
for i in range(len(columns)//3 + 1):
    name = columns[i*3]
    dicts[name] = []
    for j in range(1, df[name].count()):
        step = int(df.iloc[j].iloc[i*3])
        if j+1 < df[name].count():
            next_step = int(df.iloc[j+1].iloc[i*3])
        else:
            next_step = 2000
        variation = float(df.iloc[j].iloc[i*3 + 1])
        for k in range(step, next_step):
            dicts[name].append(variation)
xs = [i for i in range(len(dicts[name]))]
ys = []
fig = plt.figure()
ax = plt.subplot(111)

for name in dicts.keys():
    y = list(dicts[name])
    y = np.array(y)
    y = (y-min(y))/(max(y)-min(y))
    ys.append(list(y))
    ax.plot(xs, y, label=name)

mean_ys_axis = np.mean(ys, axis=0)
plt.step(xs, mean_ys_axis, label='mean', color='black', linestyle='--')
      
plt.xlabel('Iteration')
plt.ylabel('Variation (normalized)')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()