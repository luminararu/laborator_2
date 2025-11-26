import numpy as np
from skimage import data

import matplotlib.pyplot as plt
import pandas as pd
from scipy import misc, ndimage
from scipy.ndimage import median_filter

N = 32
n1 = np.arange(N)
n2 = np.arange(N)
N1, N2 = np.meshgrid(n1, n2, indexing='ij')

indice = 0
def show_image(mat, title):
    global indice
    indice = indice + 1
    plt.figure(figsize=(4,4))
    plt.imshow(mat, origin='lower', aspect='equal')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('n2 / m2')
    plt.ylabel('n1 / m1')
    plt.tight_layout()
    plt.savefig(f'fisierul_numarul_{indice}.pdf')
    plt.show()

x1 = np.sin(2*np.pi*N1 + 3*np.pi*N2)
show_image(x1, "x1(n1,n2) = sin(2π n1 + 3π n2) -domeniul timp")
x2 = np.sin(4*np.pi*N1) + np.cos(6*np.pi*N2)
show_image(x2, "x2(n1,n2) = sin(4π n1) + cos(6π n2)-domeniul timp")

Y3 = np.zeros((N,N), dtype=complex)
Y3[0,5] = 1
Y3[0, N-5] = 1

show_image(np.real(Y3), "Y3 - domeniul frecventei")
y3_spat = np.fft.ifft2(Y3)
show_image(np.real(y3_spat), "IDFT(Y3)-domeniul timp")

Y4 = np.zeros((N,N), dtype=complex)
Y4[5,0] = 1
Y4[N-5,0] = 1

show_image(np.real(Y4), "Y4 - domeniul frecventei")
y4_spat = np.fft.ifft2(Y4)
show_image(np.real(y4_spat), "IDFT(Y4)-domeniul timp")

Y5 = np.zeros((N,N), dtype=complex)
Y5[5,5] = 1
Y5[N-5,N-5] = 1

show_image(np.real(Y5), "Y5 - domeniul frecventei")
y5_spat = np.fft.ifft2(Y5)
show_image(np.real(y5_spat), "IDFT(Y5)-domeniul timp")

X = data.astronaut()   # color
X = X.mean(axis=2)     # convert to grayscale manually


plt.imshow(X, cmap='gray')
plt.show()

indice = indice + 1
Y = np.fft.fft2(X)
freq_db = 20*np.log10(np.abs(Y) + 1e-12)

plt.imshow(freq_db)
plt.title("Spectrul în dB")
plt.colorbar()
plt.savefig(f'fisierul_numarul_{indice}.pdf')
plt.show()

keep = 30
mask = np.zeros_like(Y)

mask[:keep, :keep] = 1
mask[:keep, -keep:] = 1
mask[-keep:, :keep] = 1
mask[-keep:, -keep:] = 1
Y_filtered = Y * mask
X_compressed = np.fft.ifft2(Y_filtered)
X_compressed = np.abs(X_compressed)

indice = indice + 1
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(X, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Imagine comprimată\n(atenuale frecvențe înalte)")
plt.imshow(X_compressed, cmap='gray')
plt.axis('off')
plt.savefig(f'fisierul_numarul_{indice}.pdf')

plt.show()

pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.savefig(f'fisierul_numarul_{indice}.pdf')
plt.show()

X_noisy = np.clip(X_noisy, 0, 255)
X_denoised = median_filter(X_noisy, size=5)

def calculate_snr(original, noisy):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - noisy) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

snr_noisy = calculate_snr(X, X_noisy)
snr_denoised = calculate_snr(X, X_denoised)

indice = indice + 1
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(X_noisy, cmap='gray')
plt.title(f'Cu zgomot\nSNR: {snr_noisy:.2f} dB')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(X_denoised, cmap='gray')
plt.title(f'Fără zgomot\nSNR: {snr_denoised:.2f} dB')
plt.axis('off')

plt.savefig(f'fisierul_numarul_{indice}.pdf')
plt.show()
