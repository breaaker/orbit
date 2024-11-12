'''import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化绘图
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-', animated=True)

def init():
    print('init')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

# 创建动画
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)

plt.show()'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_animation(R, z):

    # 初始化图形和轴
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r-', animated=True)
    
    R_max = np.max(R)
    z_max = np.max(z)
    z_min = np.min(z)

    def init():
        ax.set_xlim(0, R_max)
        ax.set_ylim(z_min, z_max)
        return ln,
    
    def update(frame):
        xdata.append(R[frame*10])
        ydata.append(z[frame*10])
        ln.set_data(xdata, ydata)
        return ln,
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(R)//10),
                        init_func=init, blit=True)
    
    return ani

# 使用函数创建动画
R = np.linspace(0, 1, 100)
z = np.linspace(0, 1, 100)

ani = create_animation(R, z)

# 显示动画
plt.show()