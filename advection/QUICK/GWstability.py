import numpy as np
import matplotlib.pyplot as plt

Ndt = np.linspace(0, 8, 81)
a1 = 0.5*np.ones_like(Ndt)
a2 = 0.5*np.ones_like(Ndt)
a3 = 0.5*np.ones_like(Ndt)
for i in range(len(Ndt)):
    if Ndt[i] > 2:
        a1[i] = 0.5*(1 + np.sqrt(Ndt[i]**2 - 4)/(2*Ndt[i]))
    if Ndt[i] > 1:
        a2[i] = 1 - 0.5/Ndt[i]
    if Ndt[i] > 2:
        a3[i] = 1 - 1/Ndt[i]

plt.plot(Ndt, a1, label='a1')
plt.plot(Ndt, a2, label='a2')
plt.plot(Ndt, a3, label='a3')
plt.legend()
plt.show()
