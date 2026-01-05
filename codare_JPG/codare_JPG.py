import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import color
import imageio.v2 as imageio
from scipy.fft import dctn, idctn
import imageio.v2 as imageio

X = data.astronaut()   # imagine grayscale

plt.imshow(X, cmap='gray')
plt.axis('off')
plt.show()

def rgb2ycbcr_jpeg(X):
    X = X.astype(np.float64)

    R = X[:,:,0]
    G = X[:,:,1]
    B = X[:,:,2]

    Y  =  0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
    Cr =  0.5*R - 0.418688*G - 0.081312*B + 128

    return Y, Cb, Cr


def ycbcr2rgb_jpeg(Y, Cb, Cr):
    Y = Y.astype(np.float64)
    Cb = Cb.astype(np.float64)
    Cr = Cr.astype(np.float64)

    R = Y + 1.402*(Cr - 128)
    G = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
    B = Y + 1.772*(Cb - 128)

    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    return np.stack((R, G, B), axis=2).astype(np.uint8)




X_rgb = data.astronaut()
Y, Cb, Cr= rgb2ycbcr_jpeg(X_rgb)

Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 28, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

def jpeg_channel(X, Q_jpeg):

    H, W = X.shape
    X_jpeg = np.zeros_like(X)

    # variabile care retin ceoficientii frecventelor nenuli inainte si dupa compresia
    y_nnz = 0
    y_jpeg_nnz = 0

    # Procesare pe blocuri 8x8
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = X[i:i+8, j:j+8]

            y = dctn(block, norm='ortho')

            y_q = np.round(y / Q_jpeg) * Q_jpeg
            block_jpeg = idctn(y_q, norm='ortho')

            # updatare matrice noua
            X_jpeg[i:i+8, j:j+8] = block_jpeg

            # Statistici
            y_nnz += np.count_nonzero(y)
            y_jpeg_nnz += np.count_nonzero(y_q)
    print('Componente în frecvență:', y_nnz)
    print('Componente în frecvență după cuantizare:', y_jpeg_nnz)
    return X_jpeg

Y_j  = jpeg_channel(Y, Q_jpeg)
Cb_j = jpeg_channel(Cb, Q_jpeg)
Cr_j = jpeg_channel(Cr, Q_jpeg)

X_rgb_jpeg = ycbcr2rgb_jpeg(Y_j, Cb_j, Cr_j)
X_rgb_jpeg = np.clip(X_rgb_jpeg, 0, 255).astype(np.uint8)



# Afisare rezultate
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(X, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(122)
plt.imshow(X_rgb_jpeg, cmap='gray')
plt.title('JPEG')
plt.axis('off')
plt.show()


def aplic_jpg(X, Q_nou):
    Y, Cb, Cr= rgb2ycbcr_jpeg(X)

    Y_j = jpeg_channel(Y, Q_nou)
    Cb_j = jpeg_channel(Cb, Q_nou)
    Cr_j = jpeg_channel(Cr, Q_nou)

    X_rgb_jpeg = ycbcr2rgb_jpeg(Y_j, Cb_j, Cr_j)
    X_rgb_jpeg = np.clip(X_rgb_jpeg, 0, 255).astype(np.uint8)

    return X_rgb_jpeg


'''
prag_mse = 120
factor = 1.0

while True:
    Q_nou = Q_jpeg * factor
    X_jpeg = aplic_jpg(X, Q_nou)
    mse = np.mean((X - X_jpeg) ** 2)
    if mse <= prag_mse:
        break
    factor *= 0.9


# Afisare rezultate
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(X, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(122)
plt.imshow(X_jpeg, cmap='gray')
plt.title('JPEG')
plt.axis('off')
plt.show()

'''

reader = imageio.get_reader("input.mp4", format="ffmpeg")
meta = reader.get_meta_data()
fps = meta.get('fps', 25)

writer = imageio.get_writer("output_mjpeg.mp4", fps=fps)

# salvăm primul cadru pentru afișare
frame_original = reader.get_data(0)
frame_jpeg_vis = aplic_jpg(frame_original, Q_jpeg)

# procesare video
for frame in reader:
    frame_jpeg = aplic_jpg(frame, Q_jpeg)
    writer.append_data(frame_jpeg)

reader.close()
writer.close()


plt.figure(figsize=(10,4))

plt.subplot(121)
plt.imshow(frame_original)
plt.title("Cadru original")
plt.axis("off")

plt.subplot(122)
plt.imshow(frame_jpeg_vis)
plt.title("Cadru comprimat JPEG")
plt.axis("off")

plt.show()





