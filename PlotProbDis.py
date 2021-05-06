import PlotFun as pf
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

plt.figure()
plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.8)###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.8)####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.8)###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.8)####设置上部坐标轴的粗细

delta = 2e-3
x = np.arange(0, 1, delta)

x0 = 0
y = (np.sqrt((1-x)*(1-x0)) + np.sqrt(x*x0)) ** 2
y1 = sum(y) * delta
plt.plot(x, y/y1, marker='', color='black', label='x=0')

x0 = 0.1
y = (np.sqrt((1-x)*(1-x0)) + np.sqrt(x*x0)) ** 2
y1 = sum(y) * delta
plt.plot(x, y/y1, marker='', color='blue', label='x=0.1')
plt.plot(np.ones(51, )*0.1, np.arange(0, 2.04, 0.04), linestyle='dashed', color='blue')
print(np.argmax(y))

x0 = 0.2
y = (np.sqrt((1-x)*(1-x0)) + np.sqrt(x*x0)) ** 2
y1 = sum(y) * delta
plt.plot(x, y/y1, marker='', color='green', label='x=0.2')
plt.plot(np.ones(51, )*0.2, np.arange(0, 2.04, 0.04), linestyle='dashed', color='green')
print(np.argmax(y))

x0 = 0.5
y = (np.sqrt((1-x)*(1-x0)) + np.sqrt(x*x0)) ** 2
y1 = sum(y) * delta
plt.plot(x, y/y1, marker='', color='red', label='x=0.5')
plt.plot(np.ones(51, )*0.5, np.arange(0, 2.04, 0.04), linestyle='dashed', color='red')
print(np.argmax(y))

plt.xlabel('x\'', fontdict={'family': 'Times New Roman', 'size': 20})
plt.ylabel('P(x\';x)', fontdict={'family': 'Times New Roman', 'size': 20})
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20, 'family': 'Times New Roman'}, frameon=False)
plt.axis([0, 1, 0, 2])
plt.xticks(fontproperties='Times New Roman', size=20)
plt.yticks(fontproperties='Times New Roman', size=20)
plt.show()

