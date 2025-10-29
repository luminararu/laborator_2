import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.animation import PillowWriter

N = 8
F = np.zeros((N, N), dtype=complex)
for r in range(N):
    for l in range(N):
        F[l, r] = np.exp(1j * -2 * np.pi * r * l / N)

print(F)

F_hermitian = F.conj().T
product = F @ F_hermitian
identity = N * np.eye(N)

is_unitary = np.allclose(product, identity)
print(f"\nMatricea este unitară (F @ F^H = {N} * I): {is_unitary}")

difference = product - identity
norm_difference = np.linalg.norm(difference)
print(f"Norma diferenței ||F @ F^H - {N}*I||: {norm_difference:.2e}")


fs = 100
f_signal = 5
duration = 1
N = int(fs * duration)


t = np.linspace(0, duration, N, endpoint=False)
x = np.sin(2 * np.pi * f_signal * t)

x_signal = x * np.e **(-2j * np.pi * t)


plt.plot(x_signal.real, x_signal.imag)
plt.show()
plt.savefig("semnal_simplu.pdf")
plt.close()


vec = [1, 2, 5, 7]
for rec in vec:
    x_signal = x * np.e **(-2j * np.pi * rec * t)
    plt.plot(x_signal.real, x_signal.imag)
    plt.show()
    plt.savefig(f"semnal_rotit_{rec}Hz.pdf")
    plt.close()

for rec in vec:
    x_signal = x * np.exp(-2j * np.pi * rec * t)

    # distanța față de origine
    dist = np.abs(x_signal)

    # grafic colorat în funcție de distanță
    plt.figure()
    plt.scatter(x_signal.real, x_signal.imag, c=dist, cmap='plasma')
    plt.colorbar(label='Distanța față de origine')
    plt.xlabel('Re{X}')
    plt.ylabel('Im{X}')
    plt.title(f'Frecvență de rotație = {rec} Hz')
    plt.axis('equal')
    plt.show()
    plt.savefig(f"grafic_colorat_{rec}Hz.pdf")
    plt.close()

'''

freqs = [1, 3, 5, 7]

for rec in freqs:
    x_signal = x * np.exp(-2j * np.pi * rec * t)

    fig, ax = plt.subplots()
    ax.set_xlabel('Re{X}')
    ax.set_ylabel('Im{X}')
    ax.set_title(f'Animație pentru frecvența {rec} Hz')
    ax.axis('equal')

    line, = ax.plot([], [], lw=2)
    point, = ax.plot([], [], 'ro', markersize=8)

    ax.set_xlim(x_signal.real.min() * 1.2, x_signal.real.max() * 1.2)
    ax.set_ylim(x_signal.imag.min() * 1.2, x_signal.imag.max() * 1.2)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        # Line până la frame curent
        line.set_data(x_signal.real[:frame+1], x_signal.imag[:frame+1])
        # Punctul curent (într-un array pentru set_data)
        point.set_data([x_signal.real[frame]], [x_signal.imag[frame]])
        return line, point

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=30, blit=True)

    filename = f"animatie_{rec}Hz.gif"
    ani.save(filename, writer=PillowWriter(fps=15))
    plt.close(fig)

    print(f"✅ Animația pentru {rec} Hz a fost salvată ca '{filename}'")

'''
fs = 500
T = 1
t = np.linspace(0, T, int(fs*T), endpoint=False)

f1, f2, f3 = 5, 20, 50
A1, A2, A3 = 1, 0.7, 0.5 #
x = A1*np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t) + A3*np.sin(2*np.pi*f3*t)


X = np.fft.fft(x)
N = len(x)
freq = np.fft.fftfreq(N, 1/fs)


X_mag = np.abs(X)/N


mask = freq >= 0
plt.figure(figsize=(8,4))
plt.stem(freq[mask], X_mag[mask])
plt.xlabel('Frecvența [Hz]')
plt.ylabel('|X(f)|')
plt.title('Modulul transformatei Fourier a semnalului compus')
plt.grid(True)
plt.show()
plt.savefig("transformata_fourier.pdf")
plt.close()