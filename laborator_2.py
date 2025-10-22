import numpy as np
import matplotlib.pyplot as plt

# Parametri semnal
A = 1
f = 5
fs = 2000
ts = 1 / fs

# Vectorul de eșantioane
n = np.arange(0, 1, ts)

# Semnale
a_sin = A * np.sin(2 * np.pi * f * n + np.pi/3)
a_cos = A * np.cos(2 * np.pi * f * n + np.pi/3 - np.pi/2)

# Creare subplot-uri
plt.figure(figsize=(10, 6))

# Subplot 1 - semnal sinus
plt.subplot(2, 1, 1)
plt.plot(n, a_sin, 'o-', color='blue')
plt.title('Semnal sinusoidal (a_sin)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)

# Subplot 2 - semnal cosinus
plt.subplot(2, 1, 2)
plt.plot(n, a_cos, 's-', color='red')
plt.title('Semnal cosinusoidal (a_cos)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.grid(True)

# Ajustare spațiere între subplot-uri
plt.tight_layout()
plt.show()
