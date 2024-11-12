## Boris法
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def boris(x_0, v_0, E, B, q, m, num=100):

    x = np.zeros((num, 3))
    v = np.zeros((num, 3))
    u = np.zeros(3)

    dt = 1e-9
    ##对于alpha粒子
    q_ = q * dt / (2 * m)
    
    ## 初始条件
    x[0] = x_0
    v[0] = v_0

    for i in range(1,num):
        h = B(x[i-1]) * q_
        s = 2*h/(1+np.dot(h, h))
        u = v[i-1] + q_ * E(x[i-1])
        u_ = u + np.cross((u + np.cross(u, h)), s)
        v[i] = u_ + q_ * E(x[i-1])
        x[i] = x[i-1] + v[i]*dt

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='b', linewidth=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x_max = np.max(x[:, 0])
    y_max = np.max(x[:, 1])
    z_max = np.max(x[:, 2])
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_zlim(-z_max, z_max)
    plt.show()
    R = []
    for i in range(len(x)):
        R.append(np.sqrt(x[i][0]**2+x[i][1]**2))
    ax2 = plt.figure(figsize=(8,8))
    ax2 = ax2.add_subplot(111)
    ax2.plot(R, x[:, 2], color='b', linewidth=0.8)
    ax2.set_xlabel('R')
    ax2.set_ylabel('Z')
    plt.show()
    return R, x[:, 2]

## 托卡马克磁场
def q(r):
    q1 = 1
    q2 = 0
    q3 = 0
    return q1 + q2*r + q3*r**2

def B_tokamak(vec_r, R_0, B_0):
    '''
    R_0为大半径,B_0为磁场强度
    '''
    x = vec_r[0]
    y = vec_r[1]
    z = vec_r[2]
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt((R-R_0)**2 + z**2)
    B_t = B_0*R_0/R
    B_p = B_0*r/(q(r)*R_0)

    B_x = B_t*(-y)/R - B_p*z/r*x/R
    B_y = B_t*x/R - B_p*z/r*y/R
    B_z = B_p*(R-R_0)/r
    return np.array([B_x, B_y, B_z])

def E_target(vec_r):
    return np.array([0, 0, 0])

m_p = 1.67e-27
m_alpha = 4 * m_p

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit
import qdarktheme

class orbit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Alpha Particle Orbit')
        self.layout = QVBoxLayout()
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        self.label = QLabel('Alpha Particle Orbit')
        self.layout.addWidget(self.label)

        self.layout.addWidget(QLabel('R0, m, 大半径'))
        self.R0 = QLineEdit()
        self.layout.addWidget(self.R0)
        self.R0.setText('2')

        self.layout.addWidget(QLabel('B0, T, 磁场强度'))
        self.B0 = QLineEdit()
        self.layout.addWidget(self.B0)
        self.B0.setText('5')

        self.layout.addWidget(QLabel('E0, eV, 初始能量'))
        self.E0 = QLineEdit()
        self.layout.addWidget(self.E0)
        self.E0.setText('2.3e4')

        self.layout.addWidget(QLabel('alpha, 速度各方向比例:vx,vy,vz'))
        self.alpha = QLineEdit()
        self.layout.addWidget(self.alpha)
        self.alpha.setText('-0.1,0.4,-0.2')

        self.layout.addWidget(QLabel('x0, m, 初始位置:x,y,z'))
        self.x0 = QLineEdit()
        self.layout.addWidget(self.x0)
        self.x0.setText('2.2,0,0')

        self.calbutton = QPushButton('Calculate')
        self.layout.addWidget(self.calbutton)
        self.calbutton.clicked.connect(self.cal)

        self.anibutton = QPushButton('Animation')
        self.layout.addWidget(self.anibutton)
        self.anibutton.clicked.connect(self.ani)

        self.status = QLabel()
        self.layout.addWidget(self.status)
    
    def cal(self):
        if not self.R0.text() or not self.B0.text() or not self.E0.text() or not self.alpha.text() or not self.x0.text():
            return
        R0 = float(self.R0.text())
        B0 = float(self.B0.text())
        E0 = float(self.E0.text())
        alpha = np.array([float(i) for i in self.alpha.text().split(',')])
        x0 = np.array([float(i) for i in self.x0.text().split(',')])
        self.name = f"R0={R0}, B0={B0}, E0={E0}, alpha={alpha}, x0={x0}"
        v0 = np.sqrt(2 * E0 * 1.6e-19 / m_alpha) * alpha / np.linalg.norm(alpha)
        self.R, self.z = boris(x0, v0, E_target, lambda x: B_tokamak(x, R0, B0), 2*1.6e-19, m_alpha, 50000)
        self.status.setText('Calculation Finished')

    def ani(self):
        if not hasattr(self, 'R') or not hasattr(self, 'z'):
            return
        plt.close()
        ani = create_animation(self.R, self.z)
        ani.save(f'{self.name}.gif', writer='pillow', fps=30)
        self.status.setText('Animation Saved')

def create_animation(R, z, num=30):

    # 初始化图形和轴
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r-', animated=True)
    
    R_max = np.max(R)
    R_min = np.min(R)
    z_max = np.max(z)
    z_min = np.min(z)

    def init():
        ax.set_xlim(R_min, R_max)
        ax.set_ylim(z_min, z_max)
        return ln,
    
    def update(frame):
        xdata.append(R[frame*30])
        ydata.append(z[frame*30])
        ln.set_data(xdata, ydata)
        return ln,
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(R)//num),
                        init_func=init, blit=True)
    
    return ani

app = QApplication([])
window = QMainWindow()
window.setCentralWidget(orbit())
qdarktheme.setup_theme("light")

window.show()

app.exec()