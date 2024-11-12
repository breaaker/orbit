import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = 10

x = np.linspace(0, t, N)
y = np.sin(x/4)

f = 1.5
dy = 0.1*np.sin(2*np.pi*f*x)

best_N = int(1000/(f*t))

new_y = y + dy

plt.figure(figsize=(12, 8))

#多点平均
for num in [20, best_N-10, best_N, best_N+10]:
    
    main = np.convolve(new_y, np.ones(num)/num, mode='same')

    noise = new_y - main

    plt.subplot(2, 2, [1, 2, 3, 4][[20, best_N-10, best_N, best_N+10].index(num)])
    plt.plot(x, new_y, label='Original')
    plt.plot(x, main, label='Main')
    plt.plot(x, noise, label='Noise')
    plt.plot(x, dy, label='True')
    plt.title('N=%d' % num)
    plt.legend()

plt.show()

#发现，当平均采样数为N/ft时，可以最好的去除噪声。