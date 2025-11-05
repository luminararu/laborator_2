import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import time
import librosa
from scipy.io import wavfile


# DFT prin matrice
def dft_matrix(x):
    N = len(x)
    F = np.zeros((N, N), dtype=complex)
    for r in range(N):
        for l in range(N):
            F[l, r] = np.exp(-2j * np.pi * r * l / N)
    return F @ x

# FFT recursiv
def fft_recursive(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])
    combined = [0]*N
    for k in range(N//2):
        twiddle = cmath.exp(-2j * math.pi * k / N) * odd[k]
        combined[k] = even[k] + twiddle
        combined[k + N//2] = even[k] - twiddle
    return combined

# Dimensiunile vectorilor
sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

dft_times = []
fft_times = []
numpy_fft_times = []

for N in sizes:
    x = np.random.rand(N)

    # Timp DFT
    start = time.time()
    if N <= 1024:
        dft_matrix(x)
        dft_times.append(time.time() - start)
    else:
        dft_times.append(np.nan)  # punem NaN pentru dimensiuni prea mari

    # Timp FFT recursiv
    start = time.time()
    fft_recursive(x)
    fft_times.append(time.time() - start)

    # Timp numpy FFT
    start = time.time()
    np.fft.fft(x)
    numpy_fft_times.append(time.time() - start)

# Grafic logaritmic
plt.figure(figsize=(10,6))
plt.plot(sizes, dft_times, 'o-', label='DFT (matrice)')
plt.plot(sizes, fft_times, 's-', label='FFT recursiv')
plt.plot(sizes, numpy_fft_times, '^-', label='numpy.fft.fft')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Dimensiunea vectorului N')
plt.ylabel('Timp de execuție (s)')
plt.title('Comparatie timpi de executie DFT vs FFT vs numpy.fft')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
plt.savefig("DFTvsFFT.pdf")


# ex 2

f_original = 50      # frecventa semnalului in Hz
A = 1                # amplitudinea
phi = 0              # faza
T = 1                # durata semnalului in secunde
fs_sub = 60          # frecventa de esantionare sub-Nyquist (fs < 2*f_original)

# Generare semnal continuu
t_cont = np.linspace(0, T, 5000)
signal_cont = A * np.sin(2 * np.pi * f_original * t_cont + phi)

# Eșantionare sub-Nyquist
n_samples = int(fs_sub * T)
t_sample = np.linspace(0, T, n_samples, endpoint=False)
signal_sample = A * np.sin(2 * np.pi * f_original * t_sample + phi)

# Alte două frecvențe care dau același semnal eșantionat
f_alias1 = 2 * fs_sub + f_original   # 10 Hz
f_alias2 = fs_sub + f_original   # 110 Hz

# Generare semnale continue pentru cele două frecvențe
signal_cont_alias1 = A * np.sin(2 * np.pi * f_alias1 * t_cont + phi)
signal_cont_alias2 = A * np.sin(2 * np.pi * f_alias2 * t_cont + phi)

# Eșantionare pentru cele două frecvențe
signal_sample_alias1 = A * np.sin(2 * np.pi * f_alias1 * t_sample + phi)
signal_sample_alias2 = A * np.sin(2 * np.pi * f_alias2 * t_sample + phi)

# Crearea figurilor - 4 subploturi
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Subplot 1: Doar semnalul continuu original
axes[0].plot(t_cont, signal_cont, 'b-', linewidth=1.5)
axes[0].set_ylabel('Amplitudine')
axes[0].set_title(f'Semnal continuu f={f_original}Hz')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 0.2])
axes[0].set_ylim([-1.2, 1.2])

# Subplot 2: Semnalul original cu eșantioane
axes[1].plot(t_cont, signal_cont, 'b-', linewidth=1.5)
axes[1].stem(t_sample, signal_sample, 'y', basefmt=" ",
         markerfmt='yo', linefmt='y-')
axes[1].set_ylabel('Amplitudine')
axes[1].set_title(f'Eșantionat fs={fs_sub}Hz (sub-Nyquist!)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 0.2])
axes[1].set_ylim([-1.2, 1.2])

# Subplot 3: Prima frecvență alias cu eșantioane IDENTICE
axes[2].plot(t_cont, signal_cont_alias1, color='purple', linewidth=1.5)
axes[2].stem(t_sample, signal_sample_alias1, 'y', basefmt=" ",
         markerfmt='yo', linefmt='y-')
axes[2].set_ylabel('Amplitudine')
axes[2].set_title(f'Frecvența alias f={f_alias1}Hz - EȘANTIOANE IDENTICE!')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, 0.2])
axes[2].set_ylim([-1.2, 1.2])

# Subplot 4: A doua frecvență alias cu eșantioane IDENTICE
axes[3].plot(t_cont, signal_cont_alias2, 'g-', linewidth=1.5)
axes[3].stem(t_sample, signal_sample_alias2, 'y', basefmt=" ",
         markerfmt='yo', linefmt='y-')
axes[3].set_xlabel('Timp [s]')
axes[3].set_ylabel('Amplitudine')
axes[3].set_title(f'Frecvența alias f={f_alias2}Hz - EȘANTIOANE IDENTICE!')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim([0, 0.2])
axes[3].set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.show()
plt.savefig("Exercitiul2.pdf")
# ex 3


fs_correct = 150     # fs > 2*f_original = 100 Hz

# Generare semnal continuu (pentru referință)
t_cont = np.linspace(0, T, 5000)
signal_cont = A * np.sin(2 * np.pi * f_original * t_cont + phi)

# Eșantionare CORECTĂ (peste Nyquist)
n_samples_correct = int(fs_correct * T)
t_sample_correct = np.linspace(0, T, n_samples_correct, endpoint=False)
signal_sample_correct = A * np.sin(2 * np.pi * f_original * t_sample_correct + phi)

# Celelalte două frecvențe
f_alias1 = fs_correct - f_original   # 100 Hz
f_alias2 = fs_correct + f_original   # 200 Hz

# Generare semnale continue pentru cele două frecvențe
signal_cont_alias1 = A * np.sin(2 * np.pi * f_alias1 * t_cont + phi)
signal_cont_alias2 = A * np.sin(2 * np.pi * f_alias2 * t_cont + phi)

# Eșantionare pentru cele două frecvențe
signal_sample_alias1 = A * np.sin(2 * np.pi * f_alias1 * t_sample_correct + phi)
signal_sample_alias2 = A * np.sin(2 * np.pi * f_alias2 * t_sample_correct + phi)

# Crearea figurilor
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Subplot 1: Doar semnalul continuu original
axes[0].plot(t_cont, signal_cont, 'b-', linewidth=1.5)
axes[0].set_ylabel('Amplitudine')
axes[0].set_title(f'Semnal continuu f={f_original}Hz')
axes[0].set_xlim([0, 0.5])
axes[0].set_ylim([-1.2, 1.2])

# Subplot 2: Semnalul original cu eșantioane
axes[1].plot(t_cont, signal_cont, 'b-', linewidth=1.5)
axes[1].stem(t_sample_correct, signal_sample_correct, 'y', basefmt=" ",
         markerfmt='yo', linefmt='y-')
axes[1].set_ylabel('Amplitudine')
axes[1].set_title(f'Eșantionat fs={fs_correct}Hz')
axes[1].set_xlim([0, 0.5])
axes[1].set_ylim([-1.2, 1.2])

# Subplot 3: Prima frecvență alias cu eșantioane
axes[2].plot(t_cont, signal_cont_alias1, color='purple', linewidth=1.5)
axes[2].stem(t_sample_correct, signal_sample_alias1, 'y', basefmt=" ",
         markerfmt='yo', linefmt='y-')
axes[2].set_ylabel('Amplitudine')
axes[2].set_title(f'Frecvența f={f_alias1}Hz cu eșantioane')
axes[2].set_xlim([0, 0.5])
axes[2].set_ylim([-1.2, 1.2])

# Subplot 4: A doua frecvență alias cu eșantioane
axes[3].plot(t_cont, signal_cont_alias2, 'g-', linewidth=1.5)
axes[3].stem(t_sample_correct, signal_sample_alias2, 'y', basefmt=" ",
         markerfmt='yo', linefmt='y-')
axes[3].set_xlabel('Timp [s]')
axes[3].set_ylabel('Amplitudine')
axes[3].set_title(f'Frecvența f={f_alias2}Hz cu eșantioane')
axes[3].set_xlim([0, 0.5])
axes[3].set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.show()
plt.savefig("Exercitiul3.pdf")


# ex 4: frecventa minima de esantionare este de 400


# ex 5:


# ex 6:

# (a) Citirea semnalului audio
fs, x = wavfile.read('vocale.wav')
if x.ndim > 1:
    x = x.mean(axis=1)

x = x.astype(float)
N = len(x)

# (b) Gruparea pe ferestre de 1% din semnal, cu suprapunere 50%
window_size = int(0.01 * N)
hop_size = window_size // 2          # 50% suprapunere
num_frames = (N - window_size) // hop_size + 1

# (c) Calcul FFT pentru fiecare grup
fft_matrix = []
for i in range(num_frames):
    start = i * hop_size
    frame = x[start:start + window_size]
    fft_values = np.fft.fft(frame)
    magnitude = np.abs(fft_values[:window_size // 2])  # doar partea pozitivă
    fft_matrix.append(magnitude)

# (d) Construirea matricei spectrale (fiecare coloană = FFT)
fft_matrix = np.array(fft_matrix).T  # transpunem ca fiecare coloană să fie o fereastră

# (e) Afișarea spectrogramului
plt.figure(figsize=(10, 6))
plt.imshow(fft_matrix, aspect='auto', origin='lower', cmap='jet')
plt.xlabel('Ferestre (timp)')
plt.ylabel('Frecvență (bin-uri)')
plt.title('Spectrogramă - Vocale înregistrate')
plt.colorbar(label='Amplitudine')
plt.show()
plt.savefig("Exercitiul6.pdf")

'''
#7   SNRdB = 10 log10(Psemnal/Pzgomot)
    Pzgomot, dB = Psemnal,dB - SNRdB
    Pzgomot = 90 -80 = 10

'''
