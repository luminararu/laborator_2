import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, filtfilt
from textwrap import dedent


B = 100
t = np.arange(-3, 3, 0.001)

xt = np.sinc(B * t/np.pi) ** 2

plt.figure(figsize=(6, 4))
plt.plot(t, xt)
plt.savefig('grafic1.pdf')
plt.show()

# Frecvențele de eșantionare
valori = [1, 1.5, 2, 4]

for fs in valori:
    Ts = 1 / fs

    n = np.arange(np.floor(-3 / Ts), np.ceil(3 / Ts) + 1, 1)
    tn = n * Ts
    x_n = np.sinc(B * tn/np.pi) ** 2

    t_dense = t
    x_rec = np.zeros_like(t_dense)
    for i in range(len(n)):
        x_rec += x_n[i] * np.sinc((t_dense - tn[i]) / Ts)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(t_dense, xt, label="x(t) original")
    plt.stem(tn, x_n, linefmt='r-', markerfmt='ro', basefmt='k-', label="eșantioane")
    plt.plot(t_dense, x_rec, '--', label="x̂(t) reconstruit")
    plt.savefig('grafic2.pdf')
    plt.show()

# atunci cand crestem B ul   functia tinde la functia delta dirac,  frecventa de esantinare creste
N = 100

x = np.random.rand(N)

x1 = np.convolve(x, x)
x2 = np.convolve(x1, x1)
x3 = np.convolve(x2, x2)

plt.figure(figsize=(8, 10))
plt.subplot(4,1,1); plt.plot(x); plt.title('x[n] initial')
plt.subplot(4,1,2); plt.plot(x1); plt.title('x[n]*x[n]')
plt.subplot(4,1,3); plt.plot(x2); plt.title('x[n]*x[n]*x[n]')
plt.subplot(4,1,4); plt.plot(x3); plt.title('x[n]*x[n]*x[n]*x[n]')
plt.tight_layout()
plt.savefig('grafic3.pdf')
plt.show()

# atunci fac faci convolutie functia se netezeste, pana cand tinde la destributia normala, datorita

x = np.zeros(N)
x[45:55] = 1

x1 = np.convolve(x, x)
x2 = np.convolve(x1, x1)
x3 = np.convolve(x2, x2)

plt.figure(figsize=(8, 10))
plt.subplot(4,1,1); plt.plot(x); plt.title('x[n] initial')
plt.subplot(4,1,2); plt.plot(x1); plt.title('x[n]*x[n]')
plt.subplot(4,1,3); plt.plot(x2); plt.title('x[n]*x[n]*x[n]')
plt.subplot(4,1,4); plt.plot(x3); plt.title('x[n]*x[n]*x[n]*x[n]')
plt.tight_layout()
plt.savefig('grafic4.pdf')
plt.show()


# functia devin din ce in ce neteda pana


N = 5


p = np.random.randint(-10, 11, N+1)
q = np.random.randint(-10, 11, N+1)

print("p(x) =", p)
print("q(x) =", q)

r_conv = np.convolve(p, q)
print("r(x) prin conv =", r_conv)

r_mul = np.polymul(p, q)
print("r(x) prin inmultire directa =", r_mul)


L = len(p) + len(q) - 1

P = np.fft.fft(p, n=L)
Q = np.fft.fft(q, n=L)
R_fft = np.fft.ifft(P * Q)

r_fft = np.round(np.real(R_fft)).astype(int)
print("r(x) prin FFT =", r_fft)

n = 20
d = 5

x = np.sin(2 * np.pi * np.arange(n) / n)
y = np.roll(x, d)

X = np.fft.fft(x)
Y = np.fft.fft(y)

result1 = np.fft.ifft(X * Y)
print(result1)

result2 = np.fft.ifft(Y / X)
print(result2)
#  Ambele fac corelatie dintre toate permutarile circulare ale vectorului x cu y si poate gasi coloana care da maxim, (cea care se corelaza cel mai mult)  numai ca cea cu impartire este mai rapida



def fereastra_dreptunghiulara(N, NW):
    slack = NW - N

    left_zeros_size = slack // 2
    right_zeros_size = slack - left_zeros_size

    inner = np.ones(N)
    left_zeros = np.zeros(left_zeros_size)
    right_zeros = np.zeros(right_zeros_size)
    final_vector = np.concatenate((left_zeros, inner, right_zeros))
    return final_vector

def fereastra_hanning(N, NW):
    n = np.arange(N)
    slack = NW - N

    left_zeros_size = slack // 2
    right_zeros_size = slack - left_zeros_size
    inner =  0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    left_zeros = np.zeros(left_zeros_size)
    right_zeros = np.zeros(right_zeros_size)
    final_vector = np.concatenate((left_zeros, inner, right_zeros))
    return final_vector


f = 500
A = 1
phi = 0
Nw = 200
fs = 8000

t = np.arange(Nw) / fs
sinusoida = A * np.sin(2 * np.pi * f * t + phi)

w_dreptunghi = fereastra_dreptunghiulara(100, Nw)
w_hanning = fereastra_hanning(100, Nw)

sin_dreptunghi = sinusoida * w_dreptunghi
sin_hanning = sinusoida * w_hanning


fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t, sinusoida, 'k', linewidth=2)
axes[0].set_title('Semnal Inițial (Sinusoidă f=100Hz, A=1, φ=0)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Timp (s)')
axes[0].set_ylabel('Amplitudine')
axes[0].set_ylim([-1.2, 1.2])

axes[1].plot(t, sin_dreptunghi, 'b', linewidth=2)
axes[1].set_title('Semnal după Fereastra Dreptunghiulară', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Timp (s)')
axes[1].set_ylabel('Amplitudine')
axes[1].set_ylim([-1.2, 1.2])

axes[2].plot(t, sin_hanning, 'r', linewidth=2)
axes[2].set_title('Semnal după Fereastra Hanning', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Timp (s)')
axes[2].set_ylabel('Amplitudine')
axes[2].set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('grafic5.pdf')
plt.show()


# ex 6
# usecols=2 înseamnă că iei coloana cu index 2 (a 3-a coloană)
x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, usecols=2)

# convertim la float în caz că sunt string-uri
x = x.astype(float)

signal = x[0:72]

fs = 1/3600.0
dt_seconds = 3600.0
nyquist = fs / 2.0


start_hour = 24
x = signal[start_hour : start_hour + 72]
t_x = np.arange(len(x))

plt.figure(figsize=(8,3))
plt.plot(t_x, x, label="3 zile din semnal (raw)")
plt.xlabel("Oră")
plt.ylabel("Vehicule")
plt.title("Semnal original")
plt.savefig('grafic6.pdf')
plt.show()


def moving_average(sig, w):
    sig = np.asarray(sig, dtype=float)  # ne asigurăm că e float
    return np.convolve(sig, np.ones(w), 'valid') / w


w_list = [5, 9, 13, 17]
ma_results = {w: moving_average(x, w) for w in w_list}


for w in w_list:
    plt.figure(figsize=(8,3))
    plt.plot(t_x, x, label="raw")
    plt.plot(t_x[:len(ma_results[w])], ma_results[w], label=f"MA w={w}")
    plt.title(f"Medie alunecătoare (w={w})")
    plt.xlabel("Oră")
    plt.ylabel("Vehicule")
    plt.legend()
    filename = f'grafic_MA_w_{w}.pdf'
    plt.savefig(filename)
    plt.show()



#c
fs = 1/3600.0
nyquist = fs / 2.0

cutoff_hours = 6
fc = 1 / (cutoff_hours * 3600)  # Hz
Wn = fc / nyquist

# Am ales frecventa de taiere o perioada de 6 doarece semnalul de trafic variaza lent iar fluctuatiile rapide sunt in general zgomot

#d
order = 5
rp = 5.0

b_butt, a_butt = butter(N=order, Wn=Wn, btype='low')
b_cheb, a_cheb = cheby1(N=order, rp=rp, Wn=Wn, btype='low')

x_butt = filtfilt(b_butt, a_butt, x)
x_cheb = filtfilt(b_cheb, a_cheb, x)

# Plot
plt.figure(figsize=(10,4))
plt.plot(t_x, x, label="Raw")
plt.plot(t_x, x_butt, label="Butterworth ord. 5")
plt.plot(t_x, x_cheb, label=f"Chebyshev I ord. 5, rp={rp} dB")
plt.xlabel("Oră")
plt.ylabel("Vehicule")
plt.savefig('grafic8.pdf')
plt.show()


# aleg filtrul Butterworth deaorece o sa ramana constant indiferent de parametru de rp

#6

order = 6
rp = 5.0

b_butt, a_butt = butter(N=order, Wn=Wn, btype='low')
b_cheb, a_cheb = cheby1(N=order, rp=rp, Wn=Wn, btype='low')

x_butt = filtfilt(b_butt, a_butt, x)
x_cheb = filtfilt(b_cheb, a_cheb, x)

# Plot
plt.figure(figsize=(10,4))
plt.plot(t_x, x, label="Order mai mic")
plt.plot(t_x, x_butt, color ='darkorange', label="Butterworth ord. 5")
plt.plot(t_x, x_cheb, color = 'g', label=f"Chebyshev I ord. 5, rp={rp} dB")
plt.xlabel("Oră")
plt.ylabel("Vehicule")
plt.savefig('grafic9.pdf')
plt.show()

order = 5
rp = 5.0

b_butt, a_butt = butter(N=order, Wn=Wn, btype='low')
b_cheb, a_cheb = cheby1(N=order, rp=rp, Wn=Wn, btype='low')

x_butt = filtfilt(b_butt, a_butt, x)
x_cheb = filtfilt(b_cheb, a_cheb, x)

# Plot
plt.figure(figsize=(10,4))
plt.plot(t_x, x, label="Order mai mare")
plt.plot(t_x, x_butt, color = 'darkorange', label="Butterworth ord. 5")
plt.plot(t_x, x_cheb, color = 'g', label=f"Chebyshev I ord. 5, rp={rp} dB")
plt.xlabel("Oră")
plt.ylabel("Vehicule")
plt.savefig('grafic10.pdf')
plt.show()


rp = 6.0

b_butt, a_butt = butter(N=order, Wn=Wn, btype='low')
b_cheb, a_cheb = cheby1(N=order, rp=rp, Wn=Wn, btype='low')

x_butt = filtfilt(b_butt, a_butt, x)
x_cheb = filtfilt(b_cheb, a_cheb, x)

# Plot
plt.figure(figsize=(10,4))
plt.plot(t_x, x, label="")
plt.plot(t_x, x_butt, color = 'darkorange', label="Rp mai mic")
plt.plot(t_x, x_cheb, color = 'g', label=f"Chebyshev I ord. 5, rp={rp} dB")
plt.xlabel("Oră")
plt.ylabel("Vehicule")
plt.savefig('grafic11.pdf')
plt.show()