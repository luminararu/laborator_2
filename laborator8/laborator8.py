import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

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

plt.figure(figsize=(14, 10))

plt.subplot(4,1,1)
plt.plot(t, time_series, label='Seria de timp')
plt.title('Seria de timp totală')

plt.subplot(4,1,2)
plt.plot(t, trend, color='orange', label='Trend')
plt.title('Componenta Trend')

plt.subplot(4,1,3)
plt.plot(t, season, color='green', label='Sezon')
plt.title('Componenta Sezonieră')

plt.subplot(4,1,4)
plt.plot(t, noise, color='red', label='Zgomot')
plt.title('Componenta Variabilă (Zgomot)')
plt.grid(True)
plt.tight_layout()
plt.savefig("plotul_meu1.pdf", format="pdf")
plt.show()

def autocorr_manual(x):
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    ac = []
    for lag in range(n):
        c = np.sum((x[:n-lag] - mean) * (x[lag:] - mean)) / (n * var)
        ac.append(c)
    return np.array(ac)

ac_manual = autocorr_manual(time_series)
ac_np = np.correlate(time_series - np.mean(time_series),
                     time_series - np.mean(time_series),
                     mode='full')

ac_np = ac_np[len(ac_np)//2:] / ac_np[len(ac_np)//2]

plt.figure(figsize=(8,4))
plt.plot(ac_manual, label="Manual")
plt.plot(ac_np, label="Cu functia numpy", linestyle="--")
plt.title("Autocorrelation manual si cu functie numpy")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.show()
plt.savefig("plotul_meu2.pdf", format="pdf")


p = 10
model = AutoReg(time_series, lags=p).fit()

pred = model.predict(start=p, end=N-1)

plt.figure(figsize=(10,4))
plt.plot(time_series, label="Seria originală")
plt.plot(range(p, N), pred, label="Predicție AR(p)", linestyle="--")
plt.title(f"Model AR({p}) – Seria originală și predicția")
plt.xlabel("Timp")
plt.ylabel("Valoare")
plt.legend()
plt.show()
plt.savefig("plotul_meu3.pdf", format="pdf")


p_values = range(1, 31)
m_values = [20, 50, 100]

best_p = None
best_m = None
best_score = float("inf")

for p in p_values:
    for m in m_values:
        train = time_series[:-m]
        test  = time_series[-m:]

        model = AutoReg(train, lags=p).fit()
        preds = []
        history = train.copy()
        for i in range(m):
            pred = model.predict(start=len(history), end=len(history))
            preds.append(pred[0])
            history = np.append(history, test[i])

        mse = mean_squared_error(test, preds)
        if mse < best_score:
            best_score = mse
            best_p = p
            best_m = m

print(best_p, best_m, best_score)