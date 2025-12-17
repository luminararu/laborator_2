import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
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



p = 20
train_size = int(0.8 * N)

train = time_series[:train_size]
test = time_series[train_size:]

model_ar = AutoReg(train, lags=p).fit()
pred_ar = model_ar.predict(start=train_size, end=N-1)

mse_ar = mean_squared_error(test, pred_ar)
print(f"MSE AR({p}) clasic: {mse_ar:.4f}")



p =  3
def create_lag_matrix(series, max_lag):
    X = []
    y = []
    for i in range(max_lag, len(series)):
        X.append(series[i-max_lag:i][::-1])
        y.append(series[i])
    return np.array(X), np.array(y)

X, y = create_lag_matrix(time_series, p)


selected_lags = []
remaining_lags = list(range(p))
errors = []

for step in range(p):
    best_mse = np.inf
    best_lag = None

    for lag in remaining_lags:
        candidate_lags = selected_lags + [lag]
        X_candidate = X[:, candidate_lags]

        coef = np.linalg.lstsq(X_candidate, y, rcond=None)[0]
        y_pred = X_candidate @ coef
        mse = mean_squared_error(y, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_lag = lag

    selected_lags.append(best_lag)
    remaining_lags.remove(best_lag)
    errors.append(best_mse)

print("Laguri selectate (greedy):", selected_lags)


alphas = np.linspace(0.001, 0.1, 20)
best_alpha = None
best_mse = np.inf
best_coef = None

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)

    y_pred = lasso.predict(X)
    mse = mean_squared_error(y, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_coef = lasso.coef_

print(f"Best alpha: {best_alpha}")
print("Coeficienti LASSO:", best_coef)
print("Lag-uri active:", np.where(best_coef != 0)[0])


print("\n--- COMPARATIE ---")
print(f"AR clasic MSE: {mse_ar:.4f}")
print(f"Greedy final MSE: {errors[-1]:.4f}")
print(f"LASSO MSE: {best_mse:.4f}")


def polynomial_roots(coeffs):
    coeffs = np.array(coeffs, dtype=float)

    if coeffs[0] != 1:
        coeffs = coeffs / coeffs[0]

    n = len(coeffs) - 1
    b = -coeffs[1:]

    C = np.zeros((n, n))
    C[1:, :-1] = np.eye(n - 1)
    C[:, -1] = b

    roots = np.linalg.eigvals(C)
    return roots


coeffs = [1, -6, 11, -6]

roots = polynomial_roots(coeffs)
print(roots)

coef = np.linalg.lstsq(X, y, rcond=None)[0]

eigvals = polynomial_roots(coef)
is_stationary = np.all(np.abs(eigvals) < 1)

print(is_stationary)