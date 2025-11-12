import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# frecventa de esantionare este 1/3600\

fs = 1/3600


#  1 esantion/ora = 24 esantione/zi
# 762 de zile
numar =  18288 / 24
#  fmax = fs/2     fmax = 1/7200

x = np.genfromtxt('C:\\Users\\ionut\\Downloads\\archive (2)\\Train.csv', delimiter=',', skip_header=1, usecols=2)

N = len(x)
x = x - np.mean(x)
X = np.fft.fft(x)

T = X

X = abs(X/N)
N = int(N)

X = X[:N//2]
f = fs * np.linspace(0,N//2,N//2)/N
plt.plot(f,X)
plt.yscale('log')
plt.savefig("grafic1.pdf", format='pdf')
plt.show()


#   stim ca semnalul are o componenta continua daca media este diferita de 0, iar ca sa scapam de ea pur si simple scadem media



indices = np.argsort(X)[-4:][::-1]  # indicii celor mai mari 4
frecvente_principale = f[indices]
amplitudini_principale = X[indices]

for i in range(4):
    print(f"Frecvența {i+1}: {frecvente_principale[i]:.6f} Hz, Amplitudine: {amplitudini_principale[i]:.6f}")


# Fenomenele aceste sunt cat de des se intampla, pe luna, pe saptamana, pe zi


start = 1008  # cel mai mic multiplu de 24 mai mare decât 1000
samples_per_day = 24
samples_per_month = 30 * samples_per_day  # ~30 de zile

x_luna = x[start:start + samples_per_month]
t = np.arange(len(x_luna)) / samples_per_day  # in zile

plt.figure(figsize=(12, 5))
plt.plot(t, x_luna)
plt.title("Traficul pe o lună, începând de luni")
plt.xlabel("Zile")
plt.ylabel("Număr de mașini / oră")
plt.grid(True)
plt.savefig("grafic2.pdf", format='pdf')

plt.show()


# h)  - Filtrezi semnalul astfel incat se vezi cate cicluri saptamanle exista. Vezi la ce data se termina si scazi offset ul


# am ales frevcenta pentru un interval de 3 zile
fc = 1 / (3 * 24 * 3600)
f = np.fft.fftfreq(N, d=1/fs)  # vectorul frecvențelor

# Cream o masca pentru frecventele de pastrat
mask = np.abs(f) < fc

# Aplicam filtrul în domeniul frecventei
T_filtered = T * mask

T_filtered = np.fft.ifft(T_filtered).real

# Plot: semnal original vs filtrat
plt.figure(figsize=(12,5))
plt.plot(x, label='Semnal original', alpha=0.6)
plt.plot(T_filtered, label='Semnal filtrat (trece-jos)', linewidth=2)
plt.legend()
plt.title('Filtrare în domeniul frecvenței (low-pass)')
plt.xlabel('Timp (ore)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.savefig("grafic3.pdf", format='pdf')
plt.show()



