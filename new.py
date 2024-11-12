import numpy as np
import matplotlib.pyplot as plt

I = 1     #向外为正
mu_0 = 4*np.pi

def B_0(x, y, I):
    r = np.sqrt(x**2 + y**2)
    B = (mu_0*I)/(2*np.pi*r)

    B_x = -B*y/r
    B_y = B*x/r

    return np.array([B_x, B_y])

num = 21

x = np.linspace(-1, 1, num)
y = np.linspace(-1, 1, num)
X, Y = np.meshgrid(x, y)
B = np.zeros((2, num, num))

for i in range(num):
    for j in range(num):
        x = X[i][j]
        y = Y[i][j]
        B[:, i, j] = B_0(x-0.5, y, I)
        B[:, i, j] += B_0(x+0.5, y, I)

plt.figure()
plt.streamplot(X, Y, B[0], B[1], color='b')
plt.quiver(X, Y, B[0], B[1], color='r')
plt.show()

B_mod = np.sqrt(B[0]**2 + B[1]**2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, B_mod, cmap='viridis')
plt.show()