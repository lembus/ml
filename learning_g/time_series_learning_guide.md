# Time Series Analysis & Forecasting: A Comprehensive Learning Guide

## 1. Motivation & Intuition

### Why Does This Topic Exist?
Time series analysis exists because data is often not a random snapshot of the world; it is a sequence where **ordering matters**. In standard machine learning (like predicting house prices), we often assume examples are independent and identically distributed (i.i.d.). In time series, today’s value depends heavily on yesterday’s value. The goal is to model these temporal dependencies to predict the future (forecasting) or understand the underlying structure (analysis).

### Real-World Intuition
Imagine you are a store manager ordering milk:
1. **Naive approach:** "I sold 50 bottles yesterday, so I'll order 50 for tomorrow." *(Persistence model)*
2. **Trend approach:** "Sales have been going up by 2 bottles a day for a month. I'll order 52." *(Trend modeling)*
3. **Seasonal approach:** "It's Saturday tomorrow. Every Saturday, sales double." *(Seasonality)*
4. **Correction approach:** "Last Tuesday I predicted 50 but sold 40. I over-ordered. I should be more conservative this time." *(Error correction / Moving Average)*

### Connection to Machine Learning Systems
* **Supply Chain:** E-commerce platforms predicting demand for millions of SKUs to optimize inventory placement.
* **Infrastructure:** Cloud providers predicting CPU usage to autoscale servers proactively before crashes occur.
* **Finance:** High-frequency trading algorithms predicting asset price movements based on microsecond-level history.

---

## 2. Conceptual Foundations

### A. The Components of Time Series
A time series is rarely just "one thing." We conceptually decompose it into three primary additive or multiplicative parts:
1. **Trend ($T_t$):** The long-term direction (increasing, decreasing, or flat).
2. **Seasonality ($S_t$):** Repeating patterns at fixed, known intervals (e.g., retail sales peaking every December).
3. **Residual/Noise ($\epsilon_t$):** The random, irregular variation that cannot be predicted.

### B. Stationarity: The Golden Rule
Most traditional statistical forecasting methods assume **Stationarity**. A time series is stationary if its statistical properties do not change over time:
* **Constant Mean:** The series does not drift upward or downward indefinitely.
* **Constant Variance:** The "spread" or volatility of the data is roughly consistent across time.
* **Constant Autocovariance:** The statistical relationship between time $t$ and $t-k$ depends only on the lag $k$, not on the absolute time $t$.

**Intuition:** If the underlying distribution and rules governing the data change every day (non-stationary), historical patterns cannot reliably predict future outcomes. We typically apply transformations (such as differencing or log transformations) to stabilize the mean and variance before modeling.

### C. Traditional Models (ARIMA & ETS)
* **AR (Auto-Regressive):** The current value is modeled as a weighted linear combination of *past values*. 
* **MA (Moving Average):** The current value depends on *past forecast errors* (shocks).
* **I (Integrated):** The number of differencing steps applied to the raw series to achieve stationarity.
* **Exponential Smoothing (ETS):** Models that compute forecasts using weighted averages of past observations, where recent data receives exponentially higher weight than older data.

### D. Deep Learning & Modern Approaches
While traditional models are primarily linear, deep learning architectures excel at capturing **non-linear dynamics** and complex hierarchical relationships across large datasets:
* **Recurrent Neural Networks (RNN/LSTM):** Process sequential data step-by-step maintaining a latent internal state ("memory"). While powerful, standard RNNs struggle with very long-range dependencies due to vanishing/exploding gradients.
* **Transformers (Attention Mechanisms):** Instead of sequential step-by-step recurrence, attention mechanisms look at the entire historical sequence simultaneously. They dynamically weight ("attend to") the most relevant historical time steps regardless of their temporal distance.

---

## 3. Mathematical Formulation

### A. Notation
* $y_t$: The observed value of the series at time $t$.
* $\hat{y}_{t+h|t}$: The $h$-step-ahead forecast for time $t+h$, given all historical information available up to time $t$.
* $B$: The backshift (lag) operator, defined such that $B^k y_t = y_{t-k}$.

### B. ARIMA Family
The **ARIMA(p, d, q)** model integrates autoregression, differencing, and moving averages.

#### 1. Auto-Regressive Model of Order $p$ — AR(p)

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t
$$

* $\phi_1, \dots, \phi_p$: Parameters (weights) learned from historical data.
* $c$: Constant term.
* $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$: White noise error term at time $t$.

#### 2. Moving Average Model of Order $q$ — MA(q)

$$
y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
$$

* $\theta_1, \dots, \theta_q$: Coefficients applied to past unobserved shocks (errors).

#### 3. Integration / Differencing ($d$)
To stabilize the mean, first-order differencing ($d=1$) is defined as:

$$
y'_t = (1 - B)y_t = y_t - y_{t-1}
$$

#### Full ARIMA(p, d, q) Compact Formulation

$$
(1 - \sum_{i=1}^p \phi_i B^i)(1 - B)^d y_t = c + (1 + \sum_{j=1}^q \theta_j B^j)\epsilon_t
$$

### C. Exponential Smoothing (Holt-Winters Additive Method)
For a series exhibiting both trend and seasonality with period $m$, the system updates three state equations:

$$
\text{Level:} \quad l_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + b_{t-1})
$$

$$
\text{Trend:} \quad b_t = \beta (l_t - l_{t-1}) + (1 - \beta)b_{t-1}
$$

$$
\text{Seasonal:} \quad s_t = \gamma (y_t - l_{t-1} - b_{t-1}) + (1 - \gamma)s_{t-m}
$$

$$
\text{Forecast:} \quad \hat{y}_{t+h|t} = l_t + h b_t + s_{t+h-m}
$$

* $\alpha, \beta, \gamma \in [0, 1]$: Smoothing parameters controlling the rate of state decay.

### D. Deep Learning: Scaled Dot-Product Attention
In Transformer-based forecasting models (e.g., Temporal Fusion Transformer, Informer), temporal contexts are mapped into Queries ($Q$), Keys ($K$), and Values ($V$).

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* $d_k$: The dimension of the key vectors, used as a scaling factor to prevent vanishing gradients in the softmax function during training.

---

## 4. Worked Example: Manual AR(1) Forecasting

Let us manually walk through parameter application and forecasting for a simple Auto-Regressive model of order 1.

* **Historical Data:** $y = [10, 12, 11, 14, 13]$ corresponding to $t = [1, 2, 3, 4, 5]$
* **Model Equation:** $y_t = c + \phi_1 y_{t-1} + \epsilon_t$
* **Estimated Parameters (via OLS):** $c = 2.0$, $\phi_1 = 0.8$

### Step 1: In-Sample Evaluation (Calculating Residuals)
We evaluate how well the model fits observed historical data ($\hat{y}_t = c + \phi_1 y_{t-1}$):

* **$t=2$:** $\hat{y}_2 = 2.0 + 0.8(10) = 10.0 \implies \epsilon_2 = 12 - 10.0 = +2.0$
* **$t=3$:** $\hat{y}_3 = 2.0 + 0.8(12) = 11.6 \implies \epsilon_3 = 11 - 11.6 = -0.6$
* **$t=4$:** $\hat{y}_4 = 2.0 + 0.8(11) = 10.8 \implies \epsilon_4 = 14 - 10.8 = +3.2$
* **$t=5$:** $\hat{y}_5 = 2.0 + 0.8(14) = 13.2 \implies \epsilon_5 = 13 - 13.2 = -0.2$

### Step 2: Out-of-Sample Forecasting ($t=6$)
To forecast one step into the unknown future ($h=1$), the expected error $\mathbb{E}[\epsilon_6]$ is $0$:

$$
\hat{y}_{6|5} = c + \phi_1 y_5
$$

$$
\hat{y}_{6|5} = 2.0 + 0.8(13) = 12.4
$$

### Step 3: Multi-Step Forecasting ($t=7$)
For $h=2$, we use our forecasted value $\hat{y}_{6|5}$ since the true observation $y_6$ is unknown:

$$
\hat{y}_{7|5} = c + \phi_1 \hat{y}_{6|5}
$$

$$
\hat{y}_{7|5} = 2.0 + 0.8(12.4) = 11.92
$$

---

## 5. Relevance to Machine Learning Practice

### A. Feature Engineering for Tabular Models
When applying tree-based models (e.g., XGBoost, LightGBM) to time series, raw sequential data must be transformed into a tabular supervised learning format:
1. **Lag Features:** Values at previous time steps ($y_{t-1}, y_{t-7}, y_{t-365}$).
2. **Rolling Window Statistics:** Moving averages, standard deviations, min, and max over specified window sizes (e.g., 7-day rolling mean) to capture localized momentum and volatility.
3. **Cyclical Calendar Encoding:** Standard integer encoding for temporal cycles (e.g., month 1 to 12) implies false continuity jumps between December (12) and January (1). Continuous representations use trigonometric transformations:
   $$x_{\sin} = \sin\left(\frac{2\pi t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi t}{T}\right)$$
   *(where $T$ is the period, such as 12 for months or 24 for hours).*

### B. Evaluation Metrics
* **MAE (Mean Absolute Error):** $\frac{1}{n}\sum |y_t - \hat{y}_t|$. Robust to outliers, easily interpretable in target units.
* **MAPE (Mean Absolute Percentage Error):** $\frac{100}{n}\sum \left|\frac{y_t - \hat{y}_t}{y_t}\right|$. Scale-independent percentage error. **Major Pitfall:** Asymmetric penalty (heavily penalizes over-forecasting) and approaches infinity as $y_t \to 0$.
* **MASE (Mean Absolute Scaled Error):** Scaled relative to the in-sample MAE of a naive random walk forecast:
  $$\text{MASE} = \frac{\frac{1}{h}\sum_{t=n+1}^{n+h} |y_t - \hat{y}_t|}{\frac{1}{n-1}\sum_{t=2}^n |y_t - y_{t-1}|}$$
  * $\text{MASE} < 1$: Model outperforms a naive baseline forecast.
  * $\text{MASE} > 1$: Model performs worse than simply carrying forward the last known value.

### C. Architectural Trade-offs & Selection Strategy

| Approach | Ideal Use Case | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **ARIMA / Statistical** | Univariate series, short horizon, limited historical data | Highly interpretable, fast inference, statistically rigorous bounds | Fails on cross-series patterns, struggles with high dimensionality |
| **Gradient Boosting** | Tabular setups with rich exogenous variables (promotions, weather) | Dominant performance on structured tabular data, handles missing values | Requires manual feature engineering, cannot natively extrapolate trends |
| **Deep Learning** | Massive global datasets (100k+ related series), multi-modal inputs | End-to-end representation learning, excellent long-term sequence modeling | Computationally expensive, data-hungry, difficult to debug |

---

## 6. Common Pitfalls & Misconceptions

### 1. Look-Ahead Bias (Target Leakage)
* **The Mistake:** Computing global scaling parameters (mean/std) or feature encodings across the entire dataset prior to performing temporal splitting.
* **The Fix:** Strictly enforce temporal boundaries. Compute all transformation parameters, rolling statistics, and imputations **only on the training split**, and apply those exact state parameters to the validation/test sets.

### 2. Random Cross-Validation
* **The Mistake:** Using standard $k$-fold cross-validation where future observations are randomly assigned to train alongside past test observations.
* **The Fix:** Employ **Walk-Forward Validation** (Expanding or Rolling Window cross-validation), ensuring the training set strictly precedes the validation window in chronological order.

### 3. Ignoring Naive Baselines
* **The Mistake:** Developing complex neural architectures without benchmarking against simple heuristics.
* **The Fix:** Always establish robust baselines:
  * **Naive Forecast:** $\hat{y}_{t+1} = y_t$
  * **Seasonal Naive Forecast:** $\hat{y}_{t+1} = y_{t+1-m}$

---

## 7. Interview Preparation Questions & Answers

### Foundational Questions

#### Q1: What is the difference between Time Series Analysis and Time Series Forecasting?
**Answer:**
* **Time Series Analysis** is inferential; it focuses on decomposing and understanding the structural generative mechanisms driving the data (e.g., isolating macroeconomic trends, estimating seasonal elasticity).
* **Time Series Forecasting** is predictive; it focuses on minimizing generalization error on unobserved future horizons ($t+h$), often utilizing highly complex, non-linear black-box representations where explicit structural interpretability is secondary.

#### Q2: Explain the concept of Stationarity and how to test for it.
**Answer:**
Strict stationarity requires the joint distribution of $(y_{t_1}, \dots, y_{t_k})$ to be invariant to temporal shifts. In practice, we rely on **Weak (Second-Order) Stationarity**: constant mean, constant variance, and autocovariance dependent strictly on lag distance $k$. 
* **Testing:** We use the **Augmented Dickey-Fuller (ADF)** test. The null hypothesis ($H_0$) assumes the presence of a unit root (non-stationary). A p-value below a critical threshold (typically 0.05) rejects $H_0$, indicating stationarity. Alternatively, the **KPSS test** flips the hypotheses ($H_0$ is trend-stationarity).

---

### Mathematical Questions

#### Q3: Derive the expected value and variance of a stationary AR(1) process.
**Answer:**
Let $y_t = c + \phi y_{t-1} + \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ and $|\phi| < 1$.

**Mean ($\mu$):**
Taking the expectation of both sides under stationarity ($\mathbb{E}[y_t] = \mathbb{E}[y_{t-1}] = \mu$):

$$
\mu = c + \phi \mu + \mathbb{E}[\epsilon_t] \implies \mu(1 - \phi) = c \implies \mu = \frac{c}{1 - \phi}
$$

**Variance ($\gamma_0$):**
Expressing the process in terms of deviations from the mean: $(y_t - \mu) = \phi(y_{t-1} - \mu) + \epsilon_t$.
Squaring and taking expectations:

$$
\mathbb{E}[(y_t - \mu)^2] = \phi^2 \mathbb{E}[(y_{t-1} - \mu)^2] + \mathbb{E}[\epsilon_t^2] + 2\phi \mathbb{E}[(y_{t-1}-\mu)\epsilon_t]
$$

Since $\epsilon_t$ is independent of past observations, the cross-term is $0$:

$$
\gamma_0 = \phi^2 \gamma_0 + \sigma^2 \implies \gamma_0 = \frac{\sigma^2}{1 - \phi^2}
$$

#### Q4: Why do standard Recurrent Neural Networks exhibit vanishing gradients over long temporal sequences?
**Answer:**
Let hidden state $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$. The gradient of the loss $\mathcal{L}$ at time $T$ with respect to hidden state at time $t \ll T$ involves the Jacobian chain product:
$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t+1}^T \left( \text{diag}(1 - h_k^2) W_{hh}^T \right)$$
If the singular values of the weight matrix $W_{hh}$ are strictly less than $1$, repeated matrix multiplication across long horizons ($T - t$) causes the gradient norm to decay exponentially toward zero.

---

### Applied & System Design Questions

#### Q5: Design a forecasting system for 100,000 retail product SKUs across 500 stores. Intermittent (zero-inflated) demand is prevalent.
**Answer:**
1. **Model Architecture:** Deploy a **Global Gradient Boosted Decision Tree (LightGBM)** or **Deep Autoregressive Network (DeepAR)** trained conjointly across all store-SKU combinations rather than maintaining $5 \times 10^7$ independent local models.
2. **Handling Intermittency:** Standard MSE/MAE models will predict fractional values (e.g., 0.15 units/day). Switch optimization objective to **Tweedie Deviance Loss** (which models compound Poisson-Gamma distributions natively handling zero-mass outcomes) or use two-stage **Croston’s Method** (separate models for non-zero demand probability and non-zero demand volume).
3. **Feature Engineering:** Inject hierarchical cross-sectional categorical embeddings (Store ID, Department, SKU category), rolling historical aggregate sales, promotional flags, and price elasticity lags.

#### Q6: How do you evaluate probabilistic forecasts when business operations require quantifying inventory stockout risk?
**Answer:**
Point estimates (mean/median predictions) fail to capture variance risk. We evaluate distribution predictions using:
* **Quantile Loss (Pinball Loss):** To prevent stockouts, operations may target the 95th demand percentile ($\tau = 0.95$):
  $$\mathcal{L}_\tau(y, \hat{y}) = \max\left( \tau(y - \hat{y}), (\tau - 1)(y - \hat{y}) \right)$$
* **Continuous Ranked Probability Score (CRPS):** Generalizes Mean Absolute Error to full predictive cumulative distribution functions ($F$), measuring the integrated squared difference between the forecast distribution and the empirical step function:
  $$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(z) - \mathbb{I}\{y \le z\} \right)^2 dz$$

---

### Debugging & Failure Modes

#### Q7: Your production model accurately captures historical trends but consistently misses sharp demand spikes during holiday periods. Why?
**Answer:**
* **Root Cause 1 (Loss Function Bias):** Optimizing Mean Squared Error ($\ell_2$ norm) penalizes variance, driving predictions toward the conditional conditional mean rather than capturing localized tail spikes.
* **Root Cause 2 (Feature Representation):** The model lacks explicit structural flags indicating anomalous calendar events, forcing it to treat historical holiday spikes as transient unrepeatable noise.
* **Remediation:** Introduce explicit binary indicator features (`is_black_friday`), deterministic countdown variables (`days_until_christmas`), and switch optimization to asymmetric quantile regression loss functions.

#### Q8: A newly deployed forecasting model exhibits excellent offline validation metrics. In production, its forecasts appear shifted chronologically by exactly one step ($\hat{y}_{t+1} \approx y_t$). What went wrong?
**Answer:**
This is classic **Persistence Degradation** (or the "Lag-1 Shadowing" phenomenon). The model has learned that the autoregressive signal ($y_t$) is overwhelmingly strong compared to exogenous covariates. When optimizing standard regression loss, the mathematical path of least resistance to minimize average error on highly autocorrelated series is simply echoing the most recent observation. 
* **Remediation:** Verify predictive power of exogenous variables via SHAP values. Transform the target variable by **differencing** ($\Delta y_{t+1} = y_{t+1} - y_t$), forcing the model architecture to predict structural *changes* rather than absolute levels.
