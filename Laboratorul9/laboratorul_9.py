import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings

N = 1000
t = np.arange(N)

a, b, c = 0.0005, 0.1, 5
trend = a*t**2 + b*t + c

freq1  = 50
freq2 = 200
season = 2*np.sin(2*np.pi*t/freq1) + 1.5*np.sin(2*np.pi*t/freq2)

np.random.seed(42)
noise = np.random.normal(0, 0.5, N)

time_series = trend + season + noise


alfa_valori = np.linspace(0.01, 0.99, 20)

best_alpha = 0
best_mse = 1e9

y=time_series

n=N
for alfa in alfa_valori:
    s = np.zeros(n)
    s[0] = y[0]

    for i in range(1, n):
        s[i] = alfa * y[i] + (1 - alfa) * s[i-1]

    mse = np.mean((y - s)**2)

    if mse < best_mse:
        best_alpha = alfa
        best_mse = mse
        s_simplu = s.copy()

print(best_alpha, best_mse)

plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(time_series)
plt.title("Time series originala")
plt.ylabel("Valoare")

plt.subplot(2, 1, 2)
plt.plot(s_simplu)
plt.title("Mediere exponentiala simpla")
plt.xlabel("Timp")
plt.ylabel("Valoare")

plt.tight_layout()
plt.savefig("grafic_1.pdf")
plt.show()
plt.savefig("")


best_mse = 1e9

for alpha in np.linspace(0.1, 0.9, 10):
    for beta in np.linspace(0.1, 0.9, 10):

        l = np.zeros(n)
        b = np.zeros(n)

        l[0] = y[0]
        b[0] = y[1] - y[0]

        for i in range(1, n):
            l[i] = alpha * y[i] + (1 - alpha) * (l[i-1] + b[i-1])
            b[i] = beta * (l[i] - l[i-1]) + (1 - beta) * b[i-1]

        y_hat = l + b
        mse = np.mean((y - y_hat)**2)

        if mse < best_mse:
            best_mse = mse
            best_a = alpha
            best_b = beta
            best_holt = y_hat.copy()

print("Holt: alpha =", round(best_a, 3), "beta =", round(best_b, 3))


plt.subplot(2, 1, 1)
plt.plot(time_series)
plt.title("Time series originala")
plt.ylabel("Valoare")


plt.subplot(2, 1, 2)
plt.plot(best_holt)
plt.title("Mediere exponentiala dubla")
plt.xlabel("Timp")
plt.ylabel("Valoare")

plt.tight_layout()
plt.savefig("grafic_2.pdf")
plt.show()


m = 12
best_mse = 1e9

for alpha in [0.2, 0.4, 0.6]:
    for beta in [0.2, 0.4, 0.6]:
        for gamma in [0.2, 0.4, 0.6]:

            l = np.zeros(n)
            b = np.zeros(n)
            s = np.zeros(m)

            l[0] = y[0]
            b[0] = y[1] - y[0]
            s[:] = y[:m] - np.mean(y[:m])

            y_hat = np.zeros(n)

            for i in range(1, n):
                l[i] = alpha * (y[i] - s[i % m]) + (1 - alpha) * (l[i-1] + b[i-1])
                b[i] = beta * (l[i] - l[i-1]) + (1 - beta) * b[i-1]
                s[i % m] = gamma * (y[i] - l[i]) + (1 - gamma) * s[i % m]
                y_hat[i] = l[i] + b[i] + s[i % m]

            mse = np.mean((y - y_hat)**2)

            if mse < best_mse:
                best_mse = mse
                best_hw = y_hat.copy()
                best_params = (alpha, beta, gamma)

print("Holt-Winters: alpha, beta, gamma =", best_params)

plt.subplot(2, 1, 1)
plt.plot(time_series)
plt.title("Time series originala")
plt.ylabel("Valoare")


plt.subplot(2, 1, 2)
plt.plot(best_hw)
plt.title("Mediere exponentiala dubla")
plt.xlabel("Timp")
plt.ylabel("Valoare")

plt.tight_layout()
plt.savefig("grafic_3.pdf")
plt.show()


q = 10
y = time_series
n = len(y)
medie_q = np.zeros(n)

for i in range(n):
    if i < q:
        medie_q[i] = np.mean(y[:i + 1])
    else:
        medie_q[i] = np.mean(y[i - q:i])

epsilon = np.zeros(n)

for i in range(n):
    epsilon[i] = y[i] - medie_q[i]

theta = np.ones(q) / q
ma_q = np.zeros(n)

for i in range(n):
    suma = 0
    for j in range(q):
        if i - j >= 0:
            suma += theta[j] * epsilon[i - j]
    ma_q[i] = medie_q[i] + suma

plt.figure(figsize=(8,6))

plt.subplot(3,1,1)
plt.plot(y)
plt.title("Seria originala")
plt.subplot(3,1,3)
plt.plot(ma_q)
plt.title("Model MA(" + str(q) + ")")

plt.tight_layout()
plt.savefig("grafic_4.pdf")
plt.show()


best_aic = 1e9
best_p = 0
best_q = 0


y = time_series

for p in range(21):
    for q in range(21):
        try:
            print(p, q)
            model = ARIMA(y, order=(p, 0, q), l1 = 3)
            result = model.fit()

            if result.aic < best_aic:
                best_aic = result.aic
                best_p = p
                best_q = q
        except:
            pass

print("p optim =", best_p)
print("q optim =", best_q)
print("minim =", best_aic)



