import numpy as np
import matplotlib.pyplot as plt


def hankel_matrix(time_series, l):
    N = len(time_series)
    k = N - l + 1
    h = np.zeros((l, k))
    for i in range(k):
        h[:, i] = time_series[i:i + l]
    return h


def diagonal_averaging(X):
    L, K = X.shape
    N = L + K - 1
    x_reconstructed = np.zeros(N)
    counts = np.zeros(N)

    for i in range(L):
        for j in range(K):
            x_reconstructed[i + j] += X[i, j]
            counts[i + j] += 1
    return x_reconstructed / counts


def SSA(time_series, l):
    X = hankel_matrix(time_series, l)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    components = []
    for i in range(len(S)):
        Xi = S[i] * np.outer(U[:, i], Vt[i, :])
        xi = diagonal_averaging(Xi)
        components.append(xi)

    return components, S


N = 1000
t = np.arange(N)

a, b, c = 0.0005, 0.1, 5
trend = a * t ** 2 + b * t + c

freq1 = 50
freq2 = 200
season = 2 * np.sin(2 * np.pi * t / freq1) + 1.5 * np.sin(2 * np.pi * t / freq2)

np.random.seed(42)
noise = np.random.normal(0, 0.5, N)

time_series = trend + season + noise

l = 50
components, S = SSA(time_series, l)

plt.figure(figsize=(15, 10))
plt.subplot(6, 1, 1)
plt.plot(time_series, label='Seria originală', color='black')
plt.legend()

for i in range(5):
    plt.subplot(6, 1, i + 2)
    plt.plot(components[i], label=f'Componenta {i + 1} (σ={S[i]:.2f})')
    plt.legend()

plt.tight_layout()
plt.show()


X = hankel_matrix(time_series, l)


XtX = X.T @ X
XXt = X @ X.T


U, S, Vt = np.linalg.svd(X, full_matrices=True)

print(S)
print(Vt)
print(U)


print("Valorile proprii ale lui XXT si XTX sunt:\n", S**2)

print("Vectorii proprii ai lui XXT sunt:\n",  U)

print("Vettorii proprii ai lui XTX sunt:", Vt)




