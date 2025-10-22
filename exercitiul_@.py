import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import sawtooth



A = 1
f = 5
fs = 2000
ts = 1 / fs

# Vectorul de eșantioane
n = np.arange(0, 1, ts)

# Semnale cu faze diferite
faza_1 = np.pi / 3
faza_2 = np.pi / 4
faza_3 = np.pi / 2
faza_4 = np.pi / 5

x1 = A * np.sin(2 * np.pi * f * n + faza_1)
x2 = A * np.sin(2 * np.pi * f * n + faza_2)
x3 = A * np.sin(2 * np.pi * f * n + faza_3)
x4 = A * np.sin(2 * np.pi * f * n + faza_4)

# Grafic 1: Toate semnalele cu faze diferite
plt.figure(figsize=(12, 6))
plt.plot(n, x1, label=f'φ = π/3', alpha=0.8, linewidth=1.5)
plt.plot(n, x2, label=f'φ = π/4', alpha=0.8, linewidth=1.5)
plt.plot(n, x3, label=f'φ = π/2', alpha=0.8, linewidth=1.5)
plt.plot(n, x4, label=f'φ = π/5', alpha=0.8, linewidth=1.5)
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.title('Semnale sinusoidale cu faze diferite')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.5)  # Afișăm doar primele 0.5s pentru claritate
plt.tight_layout()

x = x1

z = np.random.normal(0, 1, len(x))

SNR_values = [0.1, 1, 10, 100]


norm_x = np.linalg.norm(x)
norm_z = np.linalg.norm(z)

for i, SNR in enumerate(SNR_values):
    plt.figure(figsize=(12, 6))


    gamma = np.sqrt((norm_x ** 2) / (SNR * norm_z ** 2))

    # Semnalul cu zgomot
    x_noisy = x + gamma * z

    plt.plot(n, x_noisy, label=f'Semnal + zgomot (SNR = {SNR})', linewidth=1.5, alpha=0.8, color='red')

    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')
    plt.title(f'Semnal sinusoidal cu zgomot - SNR = {SNR}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.5)
    plt.tight_layout()

plt.show()


print("Valori γ calculate pentru fiecare SNR:")
for SNR in SNR_values:
    gamma = np.sqrt((norm_x ** 2) / (SNR * norm_z ** 2))
    print(f"SNR = {SNR:>6} => γ = {gamma:.6f}")

print(f"\nNormă semnal: ||x||₂ = {norm_x:.4f}")
print(f"Normă zgomot: ||z||₂ = {norm_z:.4f}")



F0 = 30
Fs = 44100
Ts = 1 / Fs
duration = 3

t = np.arange(0, duration, Ts)
x_square = np.sign(np.sin(2 * np.pi * F0 * t))

sd.play(x_square, Fs)
sd.wait()


# Normalizare și salvare semnal sinusoidal
sin_wave_int16 = np.int16(x_square / np.max(np.abs(x_square)) * 32767)
wavfile.write("sinusoidal.wav", fs, sin_wave_int16)

# Citire semnal salvat
fs_read, data_read = wavfile.read("sinusoidal.wav")

print("Frecvența de eșantionare citită:", fs_read)
print("Dimensiunea semnalului citit:", data_read.shape)




# Parametri generali
fs = 44100          # frecvența de eșantionare (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # vector de timp pentru 1 secundă

# Semnale
f1 = 440  # frecvența semnalului 1 (sinusoidal) - nota A4
f2 = 220  # frecvența semnalului 2 (sawtooth)

sin_wave = np.sin(2 * np.pi * f1 * t)
saw_wave = sawtooth(2 * np.pi * f2 * t)

# Redare opțională
sd.play(sin_wave, fs)
sd.wait()


# Asigurăm că semnalele au aceeași lungime
sum_wave = sin_wave + saw_wave

# Normalizare pentru a evita distorsiuni la redare
sum_wave /= np.max(np.abs(sum_wave))

# Plotare
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, sin_wave)
plt.title("Semnal sinusoidal (440 Hz)")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")

plt.subplot(3, 1, 2)
plt.plot(t, saw_wave)
plt.title("Semnal sawtooth (220 Hz)")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")

plt.subplot(3, 1, 3)
plt.plot(t, sum_wave)
plt.title("Suma celor două semnale")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")

plt.tight_layout()
plt.show()



# Parametri generali
fs = 44100          # frecvența de eșantionare (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # vector de timp pentru 1 secundă

# Semnale sinusoidale cu frecvențe diferite
f1 = 440  # 440 Hz = nota La4
f2 = 880  # 880 Hz = o octavă mai sus

sin1 = np.sin(2 * np.pi * f1 * t)
sin2 = np.sin(2 * np.pi * f2 * t)

# Le concatenăm (unul după altul)
combined = np.concatenate((sin1, sin2))

# Redare audio
sd.play(combined, fs)
sd.wait()

# Afișare grafică
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, 2, 2*fs, endpoint=False), combined)
plt.title("Două semnale sinusoidale concatenate (440 Hz urmat de 880 Hz)")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")
plt.show()

''' Se redau semnale diferite'''


fs = 8000  # Hz (poți alege orice valoare, dar 8 kHz e clar vizual)

# Vector de timp (0.01 secunde pentru claritate grafică)
t = np.linspace(0, 0.01, int(fs*0.01), endpoint=False)

# (a) f = fs / 2
f_a = fs / 2
x_a = np.sin(2 * np.pi * f_a * t)

# (b) f = fs / 4
f_b = fs / 4
x_b = np.sin(2 * np.pi * f_b * t)

# (c) f = 0 Hz
f_c = 0
x_c = np.sin(2 * np.pi * f_c * t)  # => sin(0) = 0 tot timpul


# Plotare
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x_a, 'r')
plt.title("a) Semnal sinusoidal cu f = fs/2")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")

plt.subplot(3, 1, 2)
plt.plot(t, x_b, 'g')
plt.title("b) Semnal sinusoidal cu f = fs/4")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")

plt.subplot(3, 1, 3)
plt.plot(t, x_c, 'b')
plt.title("c) Semnal sinusoidal cu f = 0 Hz")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")

plt.tight_layout()
plt.show()

# (1) Parametri generali
fs = 1000   # frecvența de eșantionare [Hz]
f = 100     # frecvența semnalului sinusoidal [Hz]
t = np.arange(0, 0.05, 1/fs)  # timp pentru 50 ms

# (2) Generăm semnalul sinusoidal
x = np.sin(2 * np.pi * f * t)

x_dec1 = x[::4]  # pornind de la primul element
t_dec1 = t[::4]

x_dec2 = x[1::4]
t_dec2 = t[1::4]

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Semnal original')
plt.title('Semnal sinusoidal original (fs = 1000 Hz)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.legend()

plt.subplot(3, 1, 2)
plt.stem(t_dec1, x_dec1)
plt.title('Semnal decimat (1/4, începând de la primul eșantion)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')

plt.subplot(3, 1, 3)
plt.stem(t_dec2, x_dec2, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title('Semnal decimat (1/4, începând de la al doilea eșantion)')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')

plt.tight_layout()
plt.show()



# Intervalul de valori pentru α
alpha = np.linspace(-np.pi/2, np.pi/2, 1000)

# Funcțiile
sin_alpha = np.sin(alpha)
linear_approx = alpha  # Aproximația simplă sin(α) ≈ α

# Aproximația Padé: (α - 7α³/60) / (1 + α²/20)
pade_approx = (alpha - (7 * alpha**3) / 60) / (1 + alpha**2 / 20)

# Calculul erorilor absolute
err_linear = np.abs(sin_alpha - linear_approx)
err_pade = np.abs(sin_alpha - pade_approx)

# === GRAFIC 1: sin(α) și aproximația liniară ===
plt.figure(figsize=(8, 4))
plt.plot(alpha, sin_alpha, label='sin(α)', linewidth=2)
plt.plot(alpha, linear_approx, '--', label='α (aprox. liniară)')
plt.title('Compararea sin(α) cu aproximația liniară')
plt.xlabel('α [rad]')
plt.ylabel('Valoare')
plt.legend()
plt.grid(True)
plt.show()

# === GRAFIC 2: eroarea pentru aproximația liniară ===
plt.figure(figsize=(8, 4))
plt.plot(alpha, err_linear, color='orange')
plt.yscale('log')  # axa Y logaritmică
plt.title('Eroarea |sin(α) - α| (scară logaritmică)')
plt.xlabel('α [rad]')
plt.ylabel('Eroare absolută (log scale)')
plt.grid(True, which='both', ls='--')
plt.show()

# === GRAFIC 3: compararea erorilor (liniară vs Padé) ===
plt.figure(figsize=(8, 4))
plt.plot(alpha, err_linear, label='|sin(α) - α|', color='orange')
plt.plot(alpha, err_pade, label='|sin(α) - Padé(α)|', color='green')
plt.yscale('log')
plt.title('Compararea erorilor pentru aproximațiile sin(α)')
plt.xlabel('α [rad]')
plt.ylabel('Eroare absolută (log scale)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()


