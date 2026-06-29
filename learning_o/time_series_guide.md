# Time Series Analysis and Forecasting for Machine Learning: A Complete Guide

---

## Part I: Foundations — What Is a Time Series and Why Does It Matter?

### 1. Motivation & Intuition

Imagine you run a small ice cream shop. Every day, you write down how many scoops you sold. After a year, you have 365 numbers, one for each day. That ordered list of numbers — where the order matters because it follows the calendar — is a **time series**.

A time series is fundamentally different from a regular dataset. In a typical ML classification problem, you might have a bag of customer records where the order doesn't matter — shuffling the rows doesn't change anything. In a time series, shuffling destroys the information. The fact that Monday's value comes before Tuesday's value *is* the information.

**Why does this matter for ML?**

Almost every real system generates time-ordered data. Revenue per quarter, server CPU usage every second, heart rate measurements from a wearable device, hourly electricity demand, daily stock prices, monthly unemployment figures — all time series. The questions we want to answer are:

- **Forecasting**: What will happen next? (How many units will we sell next month?)
- **Anomaly detection**: Is something unusual happening right now? (Is this server about to crash?)
- **Pattern understanding**: What drives the behavior? (Why do sales spike every December?)
- **Causal analysis**: Did an intervention change the trajectory? (Did our marketing campaign increase sign-ups?)

These questions underpin massive business decisions. A 1% improvement in demand forecasting at a retailer with $10B in revenue can save tens of millions of dollars in inventory costs. Accurate anomaly detection in manufacturing prevents equipment failures. Energy companies must forecast load to decide which power plants to operate.

**The core challenge**: Unlike cross-sectional data, time series data has *temporal dependencies* — what happened yesterday affects what happens today. Traditional ML methods that assume data points are independent and identically distributed (i.i.d.) break down. We need specialized tools.

### 2. Conceptual Foundations

#### 2.1 Key Terminology

**Observation (yₜ)**: A single measurement at time t. Example: "On day 5, we sold 120 scoops."

**Time index (t)**: The position in time. Can be an integer (t = 1, 2, 3, ...) or a timestamp (2025-01-15 14:30:00).

**Sampling frequency**: How often observations are recorded. Common choices: yearly, quarterly, monthly, weekly, daily, hourly, by the minute, or by the second. The frequency determines what patterns you can detect — you cannot find hourly cycles in daily data.

**Univariate vs. multivariate**: A univariate time series is a single sequence of values (just sales). A multivariate time series tracks multiple quantities simultaneously (sales, temperature, ad spend — all measured at the same times).

#### 2.2 Components of a Time Series

Most time series can be decomposed into four intuitive components:

**Trend (Tₜ)**: The long-term direction. Are sales generally going up, going down, or staying flat over years? Trend can be linear (constant growth rate) or nonlinear (accelerating growth, S-curves).

**Seasonality (Sₜ)**: Repeating patterns tied to a fixed, known period. Ice cream sales are higher in summer. Retail sales spike in December. Website traffic drops on weekends. Seasonality has a fixed period (7 days for weekly patterns, 12 months for annual patterns) and repeats predictably.

**Cyclical (Cₜ)**: Longer-term fluctuations without a fixed period. Business cycles typically last 2–10 years. Unlike seasonality, the period is variable and often unknown in advance. In practice, many texts combine trend and cycle into a single "trend-cycle" component.

**Residual / Irregular (εₜ)**: Everything left over — random noise, one-off events (a pandemic, a power outage), measurement errors. If your model is good, the residuals should look like random noise with no remaining patterns.

**Two decomposition models:**

- **Additive**: yₜ = Tₜ + Sₜ + εₜ. Use when the seasonal fluctuations are roughly constant over time (summer sales are always about 50 scoops higher than winter, regardless of overall level).
- **Multiplicative**: yₜ = Tₜ × Sₜ × εₜ. Use when seasonal fluctuations grow proportionally with the level (summer sales are 20% higher than winter — so the absolute difference grows as the business grows).

You can convert between them: taking the logarithm of a multiplicative model yields an additive model in log-space.

#### 2.3 Stationarity — The Most Important Concept

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation structure) do not change over time. Formally:

- The mean E[yₜ] = μ is constant (no trend).
- The variance Var(yₜ) = σ² is constant (no changing spread).
- The autocovariance Cov(yₜ, yₜ₊ₖ) depends only on the lag k, not on t (the correlation between values separated by k steps is the same whether you're looking at the beginning or end of the series).

**Why does stationarity matter?** Because most classical time series models (AR, MA, ARMA, ARIMA's inner workings) assume stationarity. The math that lets us estimate parameters and make forecasts relies on the idea that the future will statistically resemble the past. If the mean is drifting upward, "the past" is not representative of "the future."

**What breaks when stationarity is violated?** If you fit an AR model to a non-stationary series, your parameter estimates will be inconsistent (they don't converge to the true values even with infinite data), your confidence intervals will be wrong, and your forecasts will be unreliable.

**How to check for stationarity:**
- Visual inspection: Plot the series. Look for trends, changing variance, or shifting behavior.
- **Augmented Dickey-Fuller (ADF) test**: Tests the null hypothesis that the series has a unit root (is non-stationary). A small p-value (< 0.05) suggests stationarity.
- **KPSS test**: Tests the null hypothesis that the series *is* stationary. A small p-value suggests non-stationarity. Using both ADF and KPSS together gives a more robust picture.

**How to achieve stationarity:**
- **Differencing**: Replace yₜ with Δyₜ = yₜ − yₜ₋₁. This removes a linear trend. If the differenced series still isn't stationary, difference again (second-order differencing). Seasonal differencing: Δₛ yₜ = yₜ − yₜ₋ₛ removes seasonality of period s.
- **Log transformation**: Stabilizes variance when it grows proportionally with the level.
- **Detrending**: Fit a trend line (linear regression on time) and subtract it. Less common than differencing in practice.

#### 2.4 Autocorrelation — Measuring Temporal Dependence

The **autocorrelation function (ACF)** at lag k measures the correlation between yₜ and yₜ₋ₖ:

$$
\rho(k) = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)}
$$

This tells you how much the current value is linearly related to the value k steps ago. If ρ(1) = 0.8, today's value is strongly correlated with yesterday's. If ρ(7) = 0.6 but ρ(1) = 0.1, there's a weekly pattern.

The **partial autocorrelation function (PACF)** at lag k measures the correlation between yₜ and yₜ₋ₖ after removing the linear effects of the intermediate lags (yₜ₋₁, yₜ₋₂, ..., yₜ₋ₖ₊₁). This is like a partial regression coefficient: it isolates the direct effect of lag k after accounting for everything in between.

**Why both?** The ACF of an AR(1) process decays gradually, while its PACF cuts off sharply after lag 1. For an MA(1) process, the reverse happens. These patterns are the key diagnostic tools for identifying what model to fit.

#### 2.5 White Noise and Random Walks

**White noise** is a sequence of uncorrelated random variables with zero mean and constant variance: εₜ ~ WN(0, σ²). White noise is the "nothing left to model" baseline. If your residuals look like white noise, your model has captured all the signal.

A **random walk** is yₜ = yₜ₋₁ + εₜ. The best forecast for tomorrow is today's value. Random walks are non-stationary (the variance grows linearly with time) and are surprisingly hard to beat for many financial time series — the "efficient market hypothesis" essentially says stock prices follow a random walk.

---

## Part II: Traditional Methods

### 3. The ARIMA Family

#### 3.1 Motivation & Intuition

The ARIMA family (AutoRegressive Integrated Moving Average) is the workhorse of classical time series forecasting. The idea is beautifully simple: the current value of a time series can be explained by a combination of its own past values (autoregression), past forecast errors (moving average), and differencing to handle trends (integration).

Think of it like this. If you're trying to predict the temperature tomorrow:
- **AR thinking**: "It was 70°F today, 68°F yesterday, 65°F the day before — temperatures have been rising, so tomorrow might be around 72°F." You're using past *values* to predict.
- **MA thinking**: "My forecast yesterday was off by +3°F, and the day before by +1°F — I've been systematically under-predicting, so I should adjust upward." You're using past *errors* to correct.
- **I thinking**: "Temperatures have been trending up by about 2°F per day — let me remove that trend, model what's left, and then add the trend back." You're differencing to remove non-stationarity.

#### 3.2 Conceptual Foundations: AR(p) — Autoregressive Model

An AR(p) model says: "The current value is a linear combination of the p most recent past values, plus noise."

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t
$$

where:
- c is a constant (intercept)
- φ₁, φ₂, ..., φₚ are the autoregressive coefficients
- εₜ ~ WN(0, σ²) is white noise
- p is the order of the model (how many lags to include)

**AR(1) — the simplest case:**

$$
y_t = c + \phi_1 y_{t-1} + \varepsilon_t
$$

If φ₁ = 0.9, the series has strong positive persistence — high values tend to be followed by high values. If φ₁ = -0.5, the series oscillates. If |φ₁| ≥ 1, the series explodes (non-stationary). The **stationarity condition** for AR(1) is |φ₁| < 1. For general AR(p), all roots of the characteristic polynomial 1 − φ₁z − φ₂z² − ⋯ − φₚzᵖ = 0 must lie outside the unit circle.

**How to identify p?** Look at the PACF. For an AR(p) process, the PACF is exactly zero for all lags beyond p. The PACF "cuts off" at lag p.

#### 3.3 Conceptual Foundations: MA(q) — Moving Average Model

An MA(q) model says: "The current value is a linear combination of the q most recent forecast errors."

$$
y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
$$

where:
- μ is the mean of the series
- θ₁, θ₂, ..., θ_q are the moving average coefficients
- εₜ ~ WN(0, σ²) is white noise

**Intuition**: The MA model says the current value is the mean plus some noise, but that noise is "smoothed" by also incorporating recent past shocks. If a large positive shock hit the system 2 periods ago, the MA model remembers it and adjusts accordingly.

**MA(1) example**: yₜ = μ + εₜ + θ₁εₜ₋₁. The current value depends on today's shock and yesterday's shock. After that, the shock's influence disappears entirely. This means MA(q) processes have a "finite memory" of exactly q steps.

**How to identify q?** Look at the ACF. For an MA(q) process, the ACF is exactly zero for all lags beyond q. The ACF "cuts off" at lag q.

**The invertibility condition**: For the MA model to have a unique representation, we require that the roots of 1 + θ₁z + θ₂z² + ⋯ + θ_qzᵍ = 0 lie outside the unit circle. This is analogous to the stationarity condition for AR.

#### 3.4 ARMA(p, q) — Combining AR and MA

The ARMA model combines both:

$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}
$$

This gives more flexibility: the AR part captures persistence, the MA part captures shock propagation. ARMA requires the series to be stationary.

**Identification**: When both AR and MA components are present, both the ACF and PACF decay gradually (exponential decay, damped sine waves). This makes order selection harder — you typically compare multiple candidate (p, q) pairs using information criteria.

#### 3.5 ARIMA(p, d, q) — Adding Integration for Non-Stationary Series

Most real-world series are not stationary. ARIMA adds a differencing step:

1. Difference the series d times to make it stationary.
2. Fit an ARMA(p, q) model to the differenced series.
3. Invert the differencing to produce forecasts on the original scale.

An ARIMA(p, d, q) model applies ARMA(p, q) to the d-th difference of yₜ.

**ARIMA(0, 1, 0)** is just a random walk: Δyₜ = εₜ, or equivalently yₜ = yₜ₋₁ + εₜ.
**ARIMA(0, 1, 1)** is simple exponential smoothing (we'll see this connection later).

In practice, d = 1 handles most trends. d = 2 is sometimes needed for accelerating trends. d ≥ 3 is almost never used and usually indicates something else is wrong.

#### 3.6 Seasonal ARIMA: SARIMA(p, d, q)(P, D, Q)ₛ

Real data often has seasonality. SARIMA adds seasonal AR, differencing, and MA terms:

$$
(1 - \sum_{i=1}^{p}\phi_i B^i)(1 - \sum_{i=1}^{P}\Phi_i B^{si})(1-B)^d(1-B^s)^D y_t = (1 + \sum_{j=1}^{q}\theta_j B^j)(1 + \sum_{j=1}^{Q}\Theta_j B^{sj})\varepsilon_t
$$

where B is the backshift operator (Byₜ = yₜ₋₁), and s is the seasonal period.

**In plain language**: SARIMA(1,1,1)(1,1,1)₁₂ for monthly data means:
- AR(1) for short-term dependence
- First differencing for trend
- MA(1) for short-term shock persistence
- Seasonal AR(1) at lag 12 (this January depends on last January)
- Seasonal differencing at lag 12 (subtract last year's value)
- Seasonal MA(1) at lag 12 (this January's error relates to last January's error)

#### 3.7 Mathematical Formulation — Deriving the AR(1) Forecast

Let's derive the optimal forecast for AR(1): yₜ = c + φyₜ₋₁ + εₜ.

The h-step-ahead forecast from time T is the conditional expectation:

$$
\hat{y}_{T+h|T} = E[y_{T+h} | y_T, y_{T-1}, \ldots]
$$

For h = 1:

$$
\hat{y}_{T+1|T} = E[c + \phi y_T + \varepsilon_{T+1} | y_T, \ldots] = c + \phi y_T
$$

because E[εₜ₊₁ | past] = 0.

For h = 2:

$$
\hat{y}_{T+2|T} = E[c + \phi y_{T+1} + \varepsilon_{T+2} | y_T, \ldots] = c + \phi \hat{y}_{T+1|T} = c + \phi(c + \phi y_T) = c(1 + \phi) + \phi^2 y_T
$$

By induction, for general h:

$$
\hat{y}_{T+h|T} = c \frac{1 - \phi^h}{1 - \phi} + \phi^h y_T
$$

As h → ∞: if |φ| < 1, φʰ → 0, and the forecast converges to c/(1 − φ) = μ (the unconditional mean). This means long-horizon forecasts revert to the mean — the AR(1) model has "finite memory." The rate of reversion is controlled by |φ|: values close to 1 mean slow reversion; values close to 0 mean fast reversion.

**Forecast variance** grows with horizon:

$$
\text{Var}(\hat{y}_{T+h|T}) = \sigma^2 \frac{1 - \phi^{2h}}{1 - \phi^2}
$$

This gives us prediction intervals that widen as we forecast further ahead — reflecting increasing uncertainty.

#### 3.8 Parameter Selection — The Box-Jenkins Methodology

The Box-Jenkins approach is a systematic procedure:

**Step 1: Identify the model order.**
- Plot the series. Determine if transformation (log, Box-Cox) is needed.
- Determine d: How many differences to achieve stationarity? Use ADF test.
- Determine D: Is seasonal differencing needed?
- Examine ACF and PACF of the differenced series:
  - PACF cuts off at lag p → AR(p)
  - ACF cuts off at lag q → MA(q)
  - Both decay → ARMA (try multiple p, q)

**Step 2: Estimate parameters.**
- Maximum Likelihood Estimation (MLE) or Conditional Least Squares.
- MLE: maximize the likelihood of the observed data under the assumed model. For Gaussian errors: L(φ, θ, σ²) = ∏ₜ (1/√(2πσ²)) exp(−εₜ²/(2σ²)), where εₜ are the one-step-ahead prediction errors (innovations).

**Step 3: Diagnostic checking.**
- Examine residuals. They should be white noise:
  - No significant autocorrelation (Ljung-Box test).
  - Constant variance (plot residuals over time).
  - Approximately normal (Q-Q plot, histogram).
- If diagnostics fail, return to Step 1 and try a different model.

**Step 4: Use information criteria for model comparison.**
- **AIC** (Akaike Information Criterion): AIC = −2 log L + 2k, where k is the number of parameters.
- **BIC** (Bayesian Information Criterion): BIC = −2 log L + k log n. BIC penalizes complexity more heavily and tends to select simpler models.
- **AICc**: AIC with a small-sample correction. Preferred when n/k < 40.

Lower AIC/BIC is better. These criteria balance fit (log-likelihood) against complexity (number of parameters).

**Automated selection**: The `auto.arima` function (in R's forecast package) or `pmdarima` (Python) automates this by searching over a grid of (p, d, q)(P, D, Q) values and selecting the model with the lowest AICc.

#### 3.9 Worked Example: ARIMA Forecasting

Suppose we have monthly airline passenger data (a classic dataset) and want to forecast the next 12 months.

**Step 1**: Plot the data. We observe an upward trend and seasonality that grows over time (multiplicative). Apply log transformation: zₜ = log(yₜ). Now the seasonal amplitude looks constant (additive in log-space).

**Step 2**: Stationarity.
- First difference: Δzₜ = zₜ − zₜ₋₁ (removes trend). ADF test p-value: 0.03 — borderline.
- Seasonal difference of the first difference: ΔΔ₁₂ zₜ = Δzₜ − Δzₜ₋₁₂. ADF test p-value < 0.01. Stationary. So d = 1, D = 1.

**Step 3**: Examine ACF and PACF of ΔΔ₁₂ zₜ.
- ACF: Significant spike at lag 1 and lag 12. Suggests q = 1, Q = 1.
- PACF: Significant spike at lag 1 and lag 12. Could also suggest p = 1, P = 1.

**Step 4**: Try candidate models.
- SARIMA(0,1,1)(0,1,1)₁₂: AICc = −480.2
- SARIMA(1,1,1)(0,1,1)₁₂: AICc = −479.8 (adding AR(1) doesn't help)
- SARIMA(0,1,1)(1,1,1)₁₂: AICc = −478.5

Best model: SARIMA(0,1,1)(0,1,1)₁₂ on log-transformed data.

**Step 5**: Check residuals. Ljung-Box test p-value = 0.35 (no significant autocorrelation). Residuals look approximately normal. Good.

**Step 6**: Forecast. Produce 12-month forecasts in log-space, then exponentiate to get forecasts in original units. Include 80% and 95% prediction intervals (also exponentiated).

---

### 4. Exponential Smoothing

#### 4.1 Motivation & Intuition

Exponential smoothing is based on a simple, powerful idea: **recent observations should matter more than distant ones**. Instead of giving equal weight to all past data (like a simple average) or cutting off at p lags (like AR), exponential smoothing assigns weights that decay exponentially into the past.

Imagine you're forecasting daily sales. Common sense says yesterday's sales tell you more about tomorrow than what happened 6 months ago. Exponential smoothing formalizes this: the weight on an observation k periods ago is proportional to αᵏ, where 0 < α < 1. So yesterday gets weight α, two days ago gets weight α(1−α), three days ago gets weight α(1−α)², and so on.

#### 4.2 Simple Exponential Smoothing (SES)

For a series with no trend and no seasonality, SES forecasts are:

$$
\hat{y}_{t+1|t} = \alpha y_t + (1-\alpha) \hat{y}_{t|t-1}
$$

This is a recursive formula. The new forecast is a weighted average of the latest observation (weight α) and the previous forecast (weight 1−α).

**Unrolling the recursion:**

$$
\hat{y}_{t+1|t} = \alpha y_t + \alpha(1-\alpha) y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + \cdots
$$

The weights α, α(1−α), α(1−α)², ... sum to 1 and decay exponentially.

**The smoothing parameter α** controls how fast we forget the past:
- α close to 1: Heavy weight on the most recent observation. Forecasts react quickly to changes but are noisy.
- α close to 0: Weight spread broadly over history. Forecasts are smooth but react slowly to changes.

**Relationship to ARIMA**: SES is equivalent to ARIMA(0,1,1) with θ₁ = −(1−α). This reveals a deep connection between the two families.

#### 4.3 Holt's Linear Method (Double Exponential Smoothing)

When the series has a trend, SES will systematically lag behind. Holt's method adds a second equation for the trend:

**Level**: ℓₜ = αyₜ + (1−α)(ℓₜ₋₁ + bₜ₋₁)

**Trend**: bₜ = β*(ℓₜ − ℓₜ₋₁) + (1−β*)bₜ₋₁

**Forecast**: ŷₜ₊ₕ|ₜ = ℓₜ + h·bₜ

Here:
- ℓₜ is the estimated level (where the series is now)
- bₜ is the estimated trend (how fast it's changing)
- α controls smoothing of the level (0 < α < 1)
- β* controls smoothing of the trend (0 < β* < 1)

**Intuition for the level equation**: The new level is a weighted average of the current observation (yₜ) and the one-step-ahead forecast from the previous period (ℓₜ₋₁ + bₜ₋₁).

**Intuition for the trend equation**: The new trend estimate is a weighted average of the latest trend signal (ℓₜ − ℓₜ₋₁, how much the level just changed) and the previous trend estimate (bₜ₋₁).

**Problem**: Holt's method projects the trend indefinitely into the future. For long horizons, this often produces absurdly high or low forecasts. A common fix is the **damped trend** method:

$$
\hat{y}_{t+h|t} = \ell_t + (\phi + \phi^2 + \cdots + \phi^h) b_t
$$

where 0 < ϕ < 1 is the damping parameter. As h → ∞, the trend contribution converges to ϕ/(1−ϕ) · bₜ rather than growing without bound. The damped trend method is one of the most reliable forecasting methods across many empirical studies.

#### 4.4 Holt-Winters Method (Triple Exponential Smoothing)

When the series has both trend and seasonality, we add a third component:

**Additive Holt-Winters:**

Level: ℓₜ = α(yₜ − sₜ₋ₛ) + (1−α)(ℓₜ₋₁ + bₜ₋₁)

Trend: bₜ = β*(ℓₜ − ℓₜ₋₁) + (1−β*)bₜ₋₁

Seasonality: sₜ = γ(yₜ − ℓₜ₋₁ − bₜ₋₁) + (1−γ)sₜ₋ₛ

Forecast: ŷₜ₊ₕ|ₜ = ℓₜ + h·bₜ + sₜ₊ₕ₋ₛ

**Multiplicative Holt-Winters:**

Level: ℓₜ = α(yₜ / sₜ₋ₛ) + (1−α)(ℓₜ₋₁ + bₜ₋₁)

Trend: bₜ = β*(ℓₜ − ℓₜ₋₁) + (1−β*)bₜ₋₁

Seasonality: sₜ = γ(yₜ / (ℓₜ₋₁ + bₜ₋₁)) + (1−γ)sₜ₋ₛ

Forecast: ŷₜ₊ₕ|ₜ = (ℓₜ + h·bₜ) · sₜ₊ₕ₋ₛ

Here s is the seasonal period (e.g., 12 for monthly data with annual seasonality), and γ controls how fast the seasonal pattern updates.

**Choosing additive vs. multiplicative**: If the seasonal swings grow with the level of the series, use multiplicative. If they stay constant, use additive. When in doubt, try both and compare using holdout evaluation or information criteria.

#### 4.5 ETS Framework — A Unifying Taxonomy

The ETS (Error, Trend, Seasonality) framework organizes all exponential smoothing methods into a taxonomy:

- **Error**: Additive (A) or Multiplicative (M)
- **Trend**: None (N), Additive (A), Additive Damped (Ad), Multiplicative (M), Multiplicative Damped (Md)
- **Seasonality**: None (N), Additive (A), Multiplicative (M)

This gives 30 possible models. Each has a state-space formulation with explicit equations for the states (level, trend, seasonal components) and an observation equation. The state-space formulation allows:
- MLE for parameter estimation
- Proper likelihood-based model comparison
- Analytical formulas for prediction intervals

Example: ETS(A,A,N) = Holt's linear method with additive errors. ETS(M,Ad,M) = damped multiplicative trend with multiplicative seasonality and multiplicative errors.

---

### 5. State Space Models

#### 5.1 Motivation & Intuition

State space models provide a unified framework that encompasses both ARIMA and exponential smoothing, and extends far beyond them. The key idea is the separation of what you *observe* from the underlying *state* of the system.

Consider an airplane's position. You can't measure it perfectly — GPS has noise, radar has noise. But the airplane has a true position and velocity (the "state") that evolves smoothly over time according to physics. Your measurements are noisy glimpses of this state. A state space model formalizes this: there's a hidden state that evolves according to known dynamics, and your observations are a noisy function of that state.

In time series: the true level, trend, and seasonal components are unobserved states. Your data is a noisy observation of these states.

#### 5.2 The Linear Gaussian State Space Model

The model has two equations:

**State equation** (how the hidden state evolves):

$$
\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_{t-1} + \mathbf{B}_t \mathbf{u}_t + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)
$$

**Observation equation** (how the observation relates to the state):

$$
\mathbf{y}_t = \mathbf{H}_t \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)
$$

where:
- xₜ is the state vector (e.g., [level, trend, seasonal₁, ..., seasonalₛ])
- yₜ is the observed measurement
- Fₜ is the state transition matrix (how the state evolves)
- Hₜ is the observation matrix (how the state maps to observations)
- Bₜuₜ represents external inputs (optional control terms)
- wₜ is process noise (uncertainty in state evolution)
- vₜ is observation noise (measurement error)
- Qₜ and Rₜ are the covariance matrices of these noises

#### 5.3 The Kalman Filter

The **Kalman filter** is the optimal algorithm for estimating the state of a linear Gaussian state space model. "Optimal" means it produces the minimum mean-squared-error estimate.

The algorithm alternates between two steps:

**Predict step** (time update — project state forward):

$$
\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}_t \hat{\mathbf{x}}_{t-1|t-1}
$$

$$
\mathbf{P}_{t|t-1} = \mathbf{F}_t \mathbf{P}_{t-1|t-1} \mathbf{F}_t^\top + \mathbf{Q}_t
$$

**Update step** (measurement update — incorporate new observation):

Innovation: $\mathbf{v}_t = \mathbf{y}_t - \mathbf{H}_t \hat{\mathbf{x}}_{t|t-1}$

Innovation covariance: $\mathbf{S}_t = \mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^\top + \mathbf{R}_t$

Kalman gain: $\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^\top \mathbf{S}_t^{-1}$

Updated state: $\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \mathbf{v}_t$

Updated covariance: $\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1}$

**Intuition for the Kalman gain**: K controls how much we trust the new measurement versus our prediction.
- If observation noise R is large (unreliable measurements): K is small → we mostly trust our prediction.
- If process noise Q is large (unpredictable dynamics): K is large → we heavily weight new observations.
- If P (state uncertainty) is large: K is large → we're unsure about our state, so new data is valuable.

**Connection to exponential smoothing**: For SES, the Kalman filter reduces to exactly the exponential smoothing recursion, with α corresponding to the steady-state Kalman gain.

#### 5.4 Structural Time Series Models

Structural time series (STS) models are state space models where the state components have interpretable meaning:

$$
y_t = \mu_t + \gamma_t + \varepsilon_t
$$

**Local level** (random walk):

$$
\mu_t = \mu_{t-1} + \eta_t, \quad \eta_t \sim N(0, \sigma_\eta^2)
$$

**Local linear trend**:

$$
\mu_t = \mu_{t-1} + \nu_{t-1} + \eta_t
$$

$$
\nu_t = \nu_{t-1} + \zeta_t, \quad \zeta_t \sim N(0, \sigma_\zeta^2)
$$

**Seasonal component** (with period s):

$$
\gamma_t = -\sum_{j=1}^{s-1} \gamma_{t-j} + \omega_t, \quad \omega_t \sim N(0, \sigma_\omega^2)
$$

This constraint ensures the seasonal components sum to approximately zero over a full cycle.

**Advantages of STS over ARIMA:**
- Components are directly interpretable (you can plot the estimated trend, seasonal, and irregular separately).
- Handles missing data naturally (the Kalman filter just skips the update step when data is missing).
- Time-varying parameters: the trend can change over time, seasonality can evolve.
- Easy to incorporate external regressors, interventions, and holidays.
- Provides proper uncertainty quantification through the state covariance.

**Facebook Prophet** is essentially a structural time series model with piecewise linear trend, Fourier-based seasonality, and holiday effects, fit using Bayesian methods (Stan).

#### 5.5 Worked Example: Kalman Filter for Local Level Model

Consider the simplest structural time series: the local level model (also called the random walk plus noise model):

State: μₜ = μₜ₋₁ + ηₜ, ηₜ ~ N(0, Q)
Observation: yₜ = μₜ + εₜ, εₜ ~ N(0, R)

With Q = 1 and R = 4 (observations are noisier than state transitions).

**Initialization**: μ̂₀|₀ = 10, P₀|₀ = 1.

**Time step 1**: Observe y₁ = 12.

Predict:
- μ̂₁|₀ = μ̂₀|₀ = 10
- P₁|₀ = P₀|₀ + Q = 1 + 1 = 2

Update:
- Innovation: v₁ = y₁ − μ̂₁|₀ = 12 − 10 = 2
- Innovation variance: S₁ = P₁|₀ + R = 2 + 4 = 6
- Kalman gain: K₁ = P₁|₀ / S₁ = 2/6 = 0.333
- Updated state: μ̂₁|₁ = 10 + 0.333 × 2 = 10.667
- Updated covariance: P₁|₁ = (1 − 0.333) × 2 = 1.333

**Interpretation**: We predicted μ = 10, observed y = 12 (2 units higher), and updated our state estimate to 10.667 — moving 1/3 of the way toward the observation. We didn't move all the way because we know the observation is noisy (R = 4).

**Time step 2**: Observe y₂ = 11.

Predict:
- μ̂₂|₁ = 10.667
- P₂|₁ = 1.333 + 1 = 2.333

Update:
- v₂ = 11 − 10.667 = 0.333
- S₂ = 2.333 + 4 = 6.333
- K₂ = 2.333/6.333 = 0.368
- μ̂₂|₂ = 10.667 + 0.368 × 0.333 = 10.790
- P₂|₂ = (1 − 0.368) × 2.333 = 1.474

Notice the Kalman gain is increasing slightly because our state uncertainty grew. Over time, K converges to a steady-state value, and the filter reaches a "steady state" — this steady-state gain is precisely the smoothing parameter α of exponential smoothing.

---

## Part III: Feature Engineering for Time Series

### 6. Feature Engineering

#### 6.1 Motivation & Intuition

When you use gradient-boosted trees (XGBoost, LightGBM) or neural networks for time series forecasting, you're converting a sequential prediction problem into a tabular supervised learning problem. Instead of specialized time series models, you engineer features that capture temporal patterns and feed them into standard ML algorithms.

This approach is enormously popular in industry because:
- Tree-based models handle nonlinear relationships naturally.
- You can mix time-derived features with external features (weather, promotions, macroeconomic indicators).
- Implementation is straightforward using existing ML pipelines.
- It often wins forecasting competitions (the M5 competition was dominated by LightGBM).

The downside: you're responsible for manually encoding temporal structure that specialized models capture automatically.

#### 6.2 Lag Features

The most fundamental time series feature. A lag-k feature for observation yₜ is simply the value k periods ago: yₜ₋ₖ.

**Why it works**: If sales yesterday were high, sales today are probably high too (autocorrelation). Lag features directly capture this.

**How to implement**: For each row in your training table, add columns lag_1 = yₜ₋₁, lag_2 = yₜ₋₂, ..., lag_k = yₜ₋ₖ. The first k rows will have missing values.

**Key design decisions:**
- Which lags to include? Use domain knowledge and ACF plots. For daily data with weekly seasonality, lags 1, 7, 14, 21 are natural. For monthly data with annual seasonality, lags 1, 12, 24 are useful.
- How many lags? More lags give the model more information but increase feature dimensionality. Diminishing returns set in quickly — try starting with the dominant seasonal lag and a few short lags.

**Critical warning — data leakage**: In a forecasting context, you can only use information available at prediction time. If you're forecasting h steps ahead, you must use lags ≥ h. If you forecast 7 days ahead, the most recent available lag is lag_7, not lag_1. Using lag_1 when forecasting 7 days ahead creates leakage — the model uses future information during training and evaluation.

#### 6.3 Rolling Statistics

Compute statistics over a sliding window of recent observations:

- **Rolling mean** (moving average): Mean of the last w observations. Captures the local level.
- **Rolling standard deviation**: Std of the last w observations. Captures local volatility.
- **Rolling min/max**: Captures recent extremes.
- **Rolling median**: More robust to outliers than the mean.
- **Expanding window statistics**: Cumulative mean/std from the start of the series to the current point.

**Window size** is a tuning parameter:
- Short windows (w = 7) are responsive but noisy.
- Long windows (w = 90) are smooth but slow to adapt.
- Multiple window sizes capture different scales of temporal structure.

**Implementation note**: Use only past data in each window — the rolling statistics at time t should use observations yₜ₋ₓ, ..., yₜ₋₁ (not yₜ itself, if yₜ is the target). This avoids target leakage.

#### 6.4 Fourier Features

Seasonality of period s can be captured using sine and cosine terms at different frequencies:

$$
x_{k,t}^{(sin)} = \sin\left(\frac{2\pi k t}{s}\right), \quad x_{k,t}^{(cos)} = \cos\left(\frac{2\pi k t}{s}\right)
$$

for k = 1, 2, ..., K.

**Why Fourier terms?** Any periodic function can be approximated by a sum of sines and cosines (Fourier's theorem). Instead of using s−1 dummy variables for seasonality (one per season), you use 2K features with K << s/2. This is especially valuable for high-frequency seasonality: a daily series with hourly cycles (s = 24) needs 23 dummies but can be captured well with K = 3 or 4 Fourier pairs (6–8 features).

**Choosing K**: More terms capture finer seasonal patterns but risk overfitting. K = 1 captures a smooth, sine-wave-like seasonal pattern. K = s/2 reproduces the full seasonal pattern exactly (but then you might as well use dummy variables). In practice, K = 3–6 works for most applications. Use cross-validation to select K.

#### 6.5 Wavelet Features

While Fourier analysis decomposes a signal into global frequencies (sine/cosine waves that extend across the entire series), **wavelet transforms** provide time-frequency decomposition — they tell you which frequencies are present and *when* they occur.

**Why wavelets over Fourier?** For non-stationary signals where the frequency content changes over time. A sales series might have strong weekly seasonality that weakens during holiday periods — Fourier would average this out, but wavelets can capture it.

**Key ideas:**
- A wavelet is a short, oscillatory function (a "small wave").
- The transform slides the wavelet along the series at different scales (frequencies).
- At each position and scale, it produces a coefficient indicating how well the wavelet matches the local signal.
- Low-frequency coefficients capture trend; high-frequency coefficients capture noise and sharp changes.

**Common wavelets**: Haar (simplest, step function), Daubechies (smooth, compact), Morlet (complex, good for spectral analysis).

**Practical use as features**: Compute wavelet coefficients at several scales and use them as features in a tree-based model. This gives the model access to multi-scale temporal information.

#### 6.6 Calendar Features

Encode time-of-day, day-of-week, month-of-year, and other calendar patterns:

**Basic calendar features:**
- Hour of day (0–23)
- Day of week (0–6, where 0 = Monday)
- Day of month (1–31)
- Week of year (1–52)
- Month (1–12)
- Quarter (1–4)
- Year

**Holiday indicators:**
- Binary flag for public holidays
- Days-until and days-since nearest holiday (captures pre-holiday and post-holiday effects)
- Holiday type (Christmas vs. Labor Day behave differently)

**Event features:**
- Sporting events, school schedules, weather events
- Promotional campaigns (binary or categorical)

**Encoding strategies:**
- One-hot encoding: Standard for tree-based models.
- Cyclical encoding: For neural networks, encode cyclic features using sin/cos transformations. Day-of-week d → (sin(2πd/7), cos(2πd/7)). This tells the model that Sunday (6) is close to Monday (0), which one-hot encoding cannot represent.

---

## Part IV: Deep Learning Approaches

### 7. RNN/LSTM Variants

#### 7.1 Motivation & Intuition

Classical methods like ARIMA are linear models — they can only capture linear dependencies between past and future values. Real-world time series often have nonlinear dynamics: the effect of yesterday's stock movement on today's might depend on the current volatility regime. Deep learning models can learn these nonlinear temporal dependencies directly from data.

The most natural architecture for sequences is the **Recurrent Neural Network (RNN)**. The idea is simple: process the sequence one step at a time, maintaining a "hidden state" that summarizes everything the network has seen so far.

#### 7.2 Vanilla RNN

At each time step t, the RNN computes:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

$$
\hat{\mathbf{y}}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

where:
- xₜ is the input at time t (the current observation and any exogenous features)
- hₜ is the hidden state (a vector that "remembers" the past)
- Wₕₕ, Wₓₕ, Wₕᵧ are weight matrices (shared across all time steps)
- tanh squashes values to [−1, 1]

**The problem: vanishing gradients**. To train an RNN via backpropagation through time (BPTT), we compute gradients that flow backward through the chain of hidden states. The gradient at time t involves a product of Jacobians: ∂hₜ/∂hₜ₋₁ × ∂hₜ₋₁/∂hₜ₋₂ × ⋯. Each Jacobian has maximum singular value bounded by the spectral norm of Wₕₕ times the maximum derivative of tanh (which is 1). If this product is < 1 at each step, the gradient shrinks exponentially — the network cannot learn long-range dependencies. If > 1, the gradient explodes.

#### 7.3 LSTM (Long Short-Term Memory)

The LSTM solves the vanishing gradient problem through a gating mechanism that controls information flow. It maintains two states: the hidden state hₜ and the **cell state** cₜ (a long-term memory channel).

**Forget gate**: Decides what to discard from the cell state.

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

**Input gate**: Decides what new information to store.

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

**Candidate cell state**: New candidate values.

$$
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
$$

**Cell state update**: Combine old (gated) and new.

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

**Output gate**: Decides what to output.

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

where σ is the sigmoid function (output in [0, 1]), ⊙ is element-wise multiplication, and [hₜ₋₁, xₜ] denotes concatenation.

**Why this solves vanishing gradients**: The gradient flows through the cell state via the forget gate: ∂cₜ/∂cₜ₋₁ = fₜ. When fₜ is close to 1 (the forget gate is "open"), the gradient passes through unchanged — no shrinkage. The network can learn to keep the forget gate open for important information, allowing gradients to flow across hundreds of time steps.

**GRU (Gated Recurrent Unit)**: A simplified variant with only two gates (reset and update), merging the cell state and hidden state. Often performs comparably to LSTM with fewer parameters.

#### 7.4 Practical Considerations for RNN/LSTM in Time Series

**Input structure**: For a forecasting problem with lookback window L and forecast horizon H:
- Input: [xₜ₋ₗ₊₁, xₜ₋ₗ₊₂, ..., xₜ] — a sequence of L vectors
- Output: [ŷₜ₊₁, ŷₜ₊₂, ..., ŷₜ₊ₕ] — the next H values

**Multi-step forecasting strategies:**
- **Recursive**: Predict one step, feed the prediction back as input, repeat. Errors accumulate.
- **Direct**: Train H separate models, one per horizon. No error accumulation but inconsistencies across horizons.
- **Sequence-to-sequence**: Use an encoder-decoder architecture. The encoder processes the input sequence; the decoder generates the output sequence. The decoder can attend to encoder hidden states.

**Teacher forcing**: During training, feed the true values (not predictions) to the decoder at each step. This stabilizes training but creates a mismatch between training and inference. **Scheduled sampling** gradually transitions from teacher forcing to auto-regressive generation during training.

---

### 8. Temporal Convolutional Networks (TCN)

#### 8.1 Motivation & Intuition

TCNs apply 1D convolutions along the time axis. Why convolutions instead of recurrence?

- **Parallelism**: Convolutions process all positions simultaneously (unlike RNNs that process sequentially). This is dramatically faster on GPUs.
- **Stable gradients**: No multiplicative chain of hidden states, so no vanishing gradients.
- **Flexible receptive field**: By stacking layers with increasing dilation, TCNs can model very long-range dependencies.

#### 8.2 Key Components

**Causal convolution**: Standard convolution looks at both past and future values. For forecasting, we must ensure yₜ only depends on inputs at times ≤ t. A causal convolution shifts the filter so it only looks backward:

$$
h_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}
$$

**Dilated convolution**: A standard convolution with kernel size K has a receptive field of K. To see 1000 steps back, you'd need kernel size 1000 or 1000 layers. Dilation solves this by skipping inputs:

$$
h_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - d \cdot k}
$$

where d is the dilation rate. With K = 3 and dilation rates [1, 2, 4, 8, ...], each layer doubles the receptive field. After L layers: receptive field = (K − 1)(2ᴸ − 1) + 1. With K = 3 and L = 10 layers, the receptive field is 2 × (1024 − 1) + 1 = 2047.

**Residual connections**: Each block adds a skip connection from input to output:

$$
\text{output} = \text{ReLU}(\text{conv\_block}(x) + x)
$$

This helps gradient flow and allows very deep networks.

**The WaveNet architecture** (originally for audio generation) is a prominent example: it uses stacked dilated causal convolutions with gated activations, producing state-of-the-art results for sequential data.

---

### 9. Transformer-Based Models

#### 9.1 Motivation & Intuition

Transformers, originally designed for NLP, have revolutionized time series forecasting. Their key advantage is **self-attention**: every time step can directly attend to every other time step without going through a chain of hidden states. This enables:

- Direct modeling of long-range dependencies (no vanishing gradients, no receptive field limitations).
- Parallel computation (unlike sequential RNN processing).
- Flexible, learned attention patterns (the model decides which past time steps are most relevant for each prediction).

#### 9.2 Self-Attention for Time Series

Given input sequence X = [x₁, ..., xₜ] ∈ ℝᵀˣᵈ, self-attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

where Q = XWQ, K = XWK, V = XWV are the query, key, and value projections.

**Intuition**: Each position "queries" all other positions: "How relevant are you to my prediction?" The dot product Q·Kᵀ measures relevance. Softmax normalizes to attention weights. The output is a weighted sum of values.

**For time series**, this means: when predicting tomorrow, the model can directly attend to same-day-last-week (lag 7), or a similar pattern 3 months ago, without needing the information to propagate through a chain of hidden states.

**Multi-head attention** runs h parallel attention computations with different projections, then concatenates. Different heads can learn different temporal patterns (one head for short-term, another for seasonal, etc.).

**Positional encoding**: Transformers have no inherent notion of order. We add positional information:
- **Sinusoidal encoding**: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d)). Different frequencies for different dimensions.
- **Learnable encoding**: Train a position embedding matrix.
- **Time-specific encoding**: Encode actual timestamps (hour, day-of-week, month) rather than just position index. This is important for time series with irregular sampling.

**The O(T²) problem**: Standard self-attention has quadratic complexity in sequence length. For long time series (thousands of steps), this is prohibitive.

#### 9.3 Temporal Fusion Transformer (TFT)

The TFT is specifically designed for time series forecasting. It addresses several practical challenges that generic transformers miss.

**Key innovations:**

1. **Variable Selection Networks**: Not all input features are equally relevant. TFT uses gated residual networks (GRNs) to learn which features matter, providing interpretable feature importance scores.

2. **Static covariate encoders**: Many forecasting problems have static metadata (store ID, product category, geographic region). TFT encodes these and uses them to condition the temporal processing.

3. **Temporal processing with LSTM + attention**: TFT uses LSTMs for local temporal processing (capturing short-range patterns) and multi-head attention for long-range dependencies. This hybrid approach gets the best of both worlds.

4. **Gated skip connections**: Throughout the architecture, GRNs control information flow, allowing the model to suppress irrelevant inputs.

5. **Quantile outputs**: TFT can directly output prediction quantiles (10th, 50th, 90th percentile), providing calibrated prediction intervals.

**Architecture flow:**

Input features (known future inputs like calendar features, observed past values, static covariates) → Variable Selection → LSTM Encoder (past) / LSTM Decoder (future) → Multi-Head Attention → Output (quantile forecasts)

#### 9.4 Informer

The **Informer** tackles the O(T²) attention cost for very long sequences. Key ideas:

**ProbSparse attention**: Not all queries need full attention. Informer identifies the "dominant" queries (those with high attention entropy — they actually attend to diverse positions) and only computes full attention for those. Other queries get a default value. This reduces complexity from O(T²) to O(T log T).

**Self-attention distilling**: Between attention layers, Informer applies convolution + pooling to halve the sequence length at each layer. This creates a pyramid structure that progressively compresses the temporal representation.

**Generative decoder**: Instead of the standard auto-regressive decoder (predict one step, feed back, repeat), Informer uses a "generative" approach: the decoder takes a start token and directly generates all future predictions in one forward pass, avoiding error accumulation.

---

### 10. Hybrid Models

#### 10.1 CNN-LSTM

**Architecture**: 1D convolutional layers first extract local features (short-term patterns), then LSTM layers model the temporal dependencies among these extracted features.

**Why combine them?** CNNs are excellent at local pattern extraction (like identifying a spike pattern) but don't model temporal order well. LSTMs model temporal order well but process one step at a time. The CNN acts as a learned feature extractor, and the LSTM acts as a sequence modeler on top of the extracted features.

**Typical pipeline**:

Raw input (T × D) → Conv1D layers → Feature maps (T' × D') → LSTM layers → Hidden states → Dense layers → Forecast

**When to use**: When you expect important local patterns (specific shapes or motifs in the time series) combined with longer-range temporal dependencies.

#### 10.2 Attention Mechanisms in Hybrid Models

Attention can be added to any sequence model:

**Encoder-decoder attention**: The decoder attends to all encoder hidden states when generating each output step. This allows the model to "look back" at specific parts of the input sequence.

$$
a_{t,s} = \text{score}(h_t^{dec}, h_s^{enc})
$$

$$
c_t = \sum_s a_{t,s} h_s^{enc}
$$

$$
\hat{y}_t = f(h_t^{dec}, c_t)
$$

where aₜ,ₛ is the attention weight (how much output step t attends to input step s), and cₜ is the context vector.

**Self-attention + RNN**: Use an attention layer on top of RNN outputs. Each position's output is refined by attending to all other positions' RNN outputs. This adds global context without modifying the RNN architecture.

---

## Part V: Forecasting Evaluation

### 11. Forecasting Metrics

#### 11.1 Motivation & Intuition

How do you know if your forecast is good? You need metrics. But choosing the right metric is surprisingly tricky — different metrics reward different properties, and the "best" metric depends on the cost structure of your application.

#### 11.2 Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n}\sum_{t=1}^{n} |y_t - \hat{y}_t|
$$

**Properties:**
- In the original units (dollars, degrees, etc.) — easily interpretable.
- Robust to outliers compared to MSE (no squaring).
- All errors are weighted equally regardless of magnitude.
- The optimal forecast under MAE is the **median** of the conditional distribution, not the mean.

**When to use**: When the cost of forecast error is proportional to the magnitude (a $10 overstock and a $10 understock cost the same).

#### 11.3 Mean Absolute Percentage Error (MAPE)

$$
\text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right|
$$

**Properties:**
- Scale-independent (a percentage) — easy to compare across series.
- **Asymmetric**: Penalizes positive errors (over-prediction) less than negative errors (under-prediction). Consider: if yₜ = 100 and ŷₜ = 150, MAPE = 50%. But if yₜ = 100 and ŷₜ = 50, MAPE = 50% too. However, the optimal MAPE forecast is *biased low*.
- **Undefined when yₜ = 0** — catastrophic for intermittent demand.
- **Division by actuals inflates errors for small values**: If actual = 1 and forecast = 2, MAPE = 100%. If actual = 1000 and forecast = 1001, MAPE = 0.1%. The same absolute error gets vastly different MAPE scores.

**Symmetric MAPE (sMAPE)** attempts to fix the asymmetry:

$$
\text{sMAPE} = \frac{200}{n} \sum_{t=1}^{n} \frac{|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}
$$

but still has issues (not truly symmetric, can give misleading results).

#### 11.4 Mean Absolute Scaled Error (MASE)

$$
\text{MASE} = \frac{\frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|}{\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t - y_{t-m}|}
$$

where m is the seasonal period (m = 1 for non-seasonal data, m = 12 for monthly data with annual seasonality).

**Intuition**: The denominator is the MAE of the seasonal naïve forecast (predicting that this January will be the same as last January). A MASE < 1 means your model beats the seasonal naïve baseline. A MASE > 1 means you're doing worse than naïve.

**Properties:**
- Scale-independent.
- Symmetric.
- Well-defined for zero actuals.
- Provides a meaningful benchmark.
- Recommended by many forecasting researchers as the primary accuracy metric.

#### 11.5 Scaled Metrics and Relative Metrics

**Relative MAE (RelMAE)**: Ratio of your model's MAE to a benchmark model's MAE. RelMAE < 1 means you're better.

**Weighted MAPE (WMAPE)**: Also called MAD/Mean ratio.

$$
\text{WMAPE} = \frac{\sum_{t=1}^{n} |y_t - \hat{y}_t|}{\sum_{t=1}^{n} |y_t|}
$$

This is just the total absolute error divided by total absolute actuals. It avoids the division-by-individual-actuals problem of MAPE and gives more weight to periods with larger values (which are often more important).

#### 11.6 Probabilistic Forecast Evaluation

Point forecasts tell you the "expected" value, but decisions often depend on the range of possibilities. Probabilistic forecasts provide full predictive distributions or quantiles.

**Quantile loss (pinball loss)**: For quantile τ ∈ (0, 1):

$$
L_\tau(y, q) = \begin{cases} \tau(y - q) & \text{if } y \geq q \\ (1-\tau)(q - y) & \text{if } y < q \end{cases}
$$

This asymmetrically penalizes errors. For τ = 0.9, under-prediction is penalized 9× more than over-prediction — this makes sense because you want the 90th percentile forecast to be exceeded only 10% of the time.

**Continuous Ranked Probability Score (CRPS)**: Evaluates the entire predictive distribution against the observation.

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(x) - \mathbb{1}(x \geq y))^2 dx
$$

CRPS equals MAE when the predictive distribution is a point mass. It rewards both calibration (quantiles at the right levels) and sharpness (narrow prediction intervals).

**Calibration**: For α% prediction intervals, check if the actual value falls within the interval α% of the time. If your 90% intervals contain only 70% of observations, they're miscalibrated (too narrow).

---

## Part VI: Anomaly Detection in Time Series

### 12. Anomaly Detection

#### 12.1 Motivation & Intuition

An anomaly is an observation that doesn't conform to the expected pattern. In time series, anomalies are context-dependent: a value of 100 sales might be normal on a weekday but anomalous on Christmas Day. Similarly, a temperature of 35°C is normal in Dubai in August but anomalous in London.

Types of anomalies:
- **Point anomaly**: A single observation is unusual (spike, dip).
- **Contextual anomaly**: Unusual given the temporal context (high sales on a holiday when sales are usually low).
- **Collective anomaly**: A subsequence is unusual (a week of unusually flat activity in a normally volatile series).

#### 12.2 Seasonal-Trend Decomposition (STL)

**STL (Seasonal and Trend decomposition using Loess)** decomposes a time series into trend, seasonal, and remainder components using locally-weighted regression (loess).

**Algorithm outline:**

1. Start with an initial estimate of the trend (e.g., moving average).
2. Detrend: subtract trend from original data.
3. Estimate seasonality: for each seasonal position (e.g., each January), average across years using loess smoothing.
4. Deseasonalize: subtract seasonality from original data.
5. Estimate trend: apply loess smoothing to the deseasonalized data.
6. Iterate steps 2–5 until convergence.

**Anomaly detection via STL**: After decomposition, examine the remainder component Rₜ = yₜ − Tₜ − Sₜ. Points where |Rₜ| > k × IQR(R) (e.g., k = 3) are flagged as anomalies.

**Advantages of STL:**
- Handles any seasonal period (not just 12 or 4).
- The seasonal component can change over time (unlike classical decomposition).
- Robust to outliers (loess is a robust smoother).
- Provides interpretable decomposition.

#### 12.3 Change Point Detection

A **change point** is a time when the statistical properties of the series change abruptly. Examples: a new product launch changes the growth rate, a policy change shifts the mean, equipment degradation increases variance.

**CUSUM (Cumulative Sum)**: Tracks the cumulative sum of deviations from a reference value:

$$
S_t = \max(0, S_{t-1} + (x_t - \mu_0 - k))
$$

When Sₜ exceeds a threshold h, a change point is signaled. k is the "allowance" (sensitivity parameter) and h is the "decision interval" (detection threshold). Smaller k and h mean faster detection but more false alarms.

**Bayesian Online Change Point Detection (BOCPD)**: At each time step, computes the posterior probability of a change point having occurred at each past time step. Uses the "run length" (time since last change point) as a latent variable and updates it online using Bayes' rule.

**PELT (Pruned Exact Linear Time)**: An exact method for finding the optimal placement of multiple change points by minimizing a penalized cost function:

$$
\sum_{i=1}^{m+1} [C(y_{\tau_{i-1}+1:\tau_i})] + \beta m
$$

where C is a segment cost (e.g., negative log-likelihood), τᵢ are change point locations, m is the number of change points, and β is a penalty for model complexity. The "pruning" step makes it linear time in practice despite searching over an exponential space.

---

## Part VII: Multiple Time Series

### 13. Global Models and Hierarchical Forecasting

#### 13.1 Motivation & Intuition

In practice, you rarely forecast a single time series. A retailer might forecast demand for 100,000 products across 500 stores — that's 50 million individual time series. Fitting a separate model to each series is:

- **Data-inefficient**: Many series have limited history (new products, new stores). A single-series model can't generalize from other similar series.
- **Computationally expensive**: 50 million models to maintain.
- **Missing cross-series patterns**: Product A's sales might predict Product B's demand (substitution effects).

**Global models** fit a single model to all series simultaneously, learning shared patterns while allowing for series-specific behavior.

#### 13.2 Global Models

A **global model** (also called a cross-learning model) pools data from all time series and learns a single set of parameters.

**How to set up a global model:**

1. Stack all time series into one large dataset.
2. Add a series identifier as a feature (or use embeddings for it).
3. Include time series-specific features (product category, store size, etc.).
4. Train a single model (e.g., LightGBM, LSTM) on the pooled data.

**Why this works**: Patterns are shared. Many products show similar weekly seasonality, similar holiday effects, similar price elasticity. A global model pools statistical strength across series — it can learn these shared patterns from the entire dataset and apply them even to series with short history.

**The bias-variance trade-off:**
- **Local model** (one model per series): Low bias (perfectly adapts to each series), high variance (limited data per model).
- **Global model**: Potentially higher bias (assumes some patterns are shared), much lower variance (trained on orders of magnitude more data).

For short or noisy series, global models typically win decisively. For long, well-behaved individual series, local models might still be competitive.

**Practical implementations:**
- **LightGBM/XGBoost global model**: Very popular in industry and competitions. Use lag features, rolling statistics, and calendar features as described in Part III, pooled across all series.
- **DeepAR** (Amazon): A global autoregressive RNN model that generates probabilistic forecasts. Uses an LSTM with a learned likelihood function. Trained on all series jointly with series-specific scaling.
- **N-BEATS**: A deep residual architecture with backward and forward residual links. Operates on individual series but can be trained globally. Decomposition into trend and seasonal components is learned, not prescribed.
- **N-HiTS**: Extension of N-BEATS with multi-rate sampling for handling different temporal scales efficiently.

#### 13.3 Hierarchical Forecasting

Many time series have a hierarchical structure:

```
Total company revenue
├── Region A
│   ├── Store 1
│   │   ├── Product X
│   │   └── Product Y
│   └── Store 2
├── Region B
│   └── Store 3
```

The challenge: forecasts at different levels must be **coherent** — the forecasts for Store 1 and Store 2 should sum to the forecast for Region A. If you forecast each level independently, this won't happen automatically.

**Three approaches:**

**Top-down**: Forecast at the top level, then disaggregate using historical proportions.
- Pro: Aggregated data is smoother, easier to forecast.
- Con: Loses bottom-level detail. Proportions might not be stable.

**Bottom-up**: Forecast at the bottom level, then sum up.
- Pro: Captures individual series behavior.
- Con: Bottom-level data is noisy. Many short or sparse series.

**Optimal reconciliation (MinT)**: Forecast all levels independently, then reconcile.

Let ŷ be the vector of independent forecasts at all levels, and S be the summing matrix that maps bottom-level to all levels. The reconciled forecast is:

$$
\tilde{\mathbf{y}} = S(S^\top W^{-1} S)^{-1} S^\top W^{-1} \hat{\mathbf{y}}
$$

where W is a weight matrix (often the covariance of forecast errors).

**Intuition**: This finds the bottom-level forecasts that, when summed up, are closest to the independently-produced forecasts at all levels, weighted by forecast accuracy at each level. Levels with more accurate forecasts get more influence.

**ERM (Empirical Risk Minimization) reconciliation**: Instead of minimizing a closed-form objective, train a reconciliation model that learns how to combine forecasts across levels using holdout data.

#### 13.4 Worked Example: Hierarchical Reconciliation

Consider a company with 2 products. Independent forecasts:

- Total company: ŷ_total = 100
- Product A: ŷ_A = 55
- Product B: ŷ_B = 60

Problem: 55 + 60 = 115 ≠ 100. Incoherent.

The summing matrix S maps bottom level [A, B] to all levels:

S = [[1, 1], [1, 0], [0, 1]] (rows: total, A, B)

If we use OLS reconciliation (W = I):

Bottom-level reconciled forecasts solve: minimize ||ŷ − S·b||²

This gives b = (SᵀS)⁻¹Sᵀŷ = [[2,1,0],[1,0,1]]⁻¹ × [[1,1],[1,0],[0,1]]ᵀ × [100, 55, 60]ᵀ

SᵀS = [[2, 1],[1, 2]], (SᵀS)⁻¹ = [[2/3, -1/3],[-1/3, 2/3]]

Sᵀŷ = [1×100+1×55+0×60, 1×100+0×55+1×60] = [155, 160]

b = [[2/3, -1/3],[-1/3, 2/3]] × [155, 160] = [(310-160)/3, (-155+320)/3] = [50, 55]

Reconciled: Product A = 50, Product B = 55, Total = 105.

These are coherent (50 + 55 = 105) and closer to the original forecasts than any other coherent solution.

---

## Part VIII: Relevance to Machine Learning Practice

### 14. Where Time Series Methods Fit in ML Systems

#### 14.1 During Training

- **Temporal cross-validation**: Never use random train/test splits. Always use temporal splits: train on past, validate on future. Walk-forward validation creates multiple train/test splits by sliding the cutoff date forward.
- **Feature engineering pipelines**: Must be time-aware. Rolling statistics, lag features, and calendar features must be computed only from past data at each point.
- **Hyperparameter tuning**: Use time-based validation (not random k-fold) for hyperparameter selection.
- **Handling multiple seasonalities**: Daily data might have weekly (period 7) and annual (period 365.25) seasonality. SARIMA can only handle one; Fourier features or structural time series can handle multiple.

#### 14.2 During Inference / Production

- **Real-time vs. batch**: Some applications need real-time forecasts (algorithmic trading); others need batch forecasts (weekly demand planning).
- **Cold start**: New products/users have no history. Global models with metadata features handle this; local models cannot.
- **Concept drift**: The data-generating process changes over time (seasonality shifts, trends change). Models need periodic retraining or online adaptation.
- **Missing data**: Production data often has gaps. State space models handle this naturally; feature-based approaches need imputation.

#### 14.3 During Evaluation / Monitoring

- **Forecast monitoring**: Track forecast accuracy over time. Increasing MASE might signal model degradation or concept drift.
- **Calibration monitoring**: For probabilistic forecasts, track coverage of prediction intervals.
- **Automated retraining triggers**: Retrain when accuracy drops below a threshold or when a change point is detected.

#### 14.4 Trade-offs Summary

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| ARIMA/SARIMA | Well-understood theory, interpretable, fast, good prediction intervals | Linear, single-series, struggles with multiple seasonalities, manual tuning |
| Exponential Smoothing (ETS) | Simple, fast, surprisingly competitive, good for short series | Linear, single-series, limited to predefined patterns |
| State Space (Kalman) | Handles missing data, time-varying parameters, interpretable components | Assumes linearity (extended/unscented variants for nonlinearity), complex implementation |
| Feature-based (XGBoost/LightGBM) | Handles nonlinearity, easy to add external features, fast, scalable | Requires manual feature engineering, no native uncertainty, temporal cross-validation needed |
| RNN/LSTM | Models nonlinear dynamics, learns features automatically | Slow to train, vanishing gradients (less with LSTM), needs lots of data |
| Transformer (TFT) | Long-range dependencies, interpretable attention, handles multiple inputs | Computationally expensive, data-hungry, complex architecture |
| TCN | Parallel training, long receptive field, stable gradients | Fixed receptive field (must be set before training), less interpretable |
| Global models (DeepAR, N-BEATS) | Shares info across series, handles cold start, scales to millions of series | Might underfit unique patterns, complex infrastructure |

---

## Part IX: Common Pitfalls & Misconceptions

### 15. Pitfalls

**1. Using future information (data leakage).**
The #1 mistake in time series ML. Examples: using lag_1 when forecasting 7 steps ahead; computing rolling statistics that include the target value; using features derived from the test period during training. Always ask: "At prediction time, would I actually have this information?"

**2. Random train/test splits.**
Random splits destroy temporal structure and dramatically overestimate accuracy. A data point from 2023 might end up in training while its neighbors from 2022 and 2024 are in the test set — the model "sees the future."

**3. Confusing stationarity with "not changing."**
A stationary series can change a lot — it just fluctuates around a constant mean. A series can trend upward yet have stationary *first differences*. Stationarity is about statistical *properties* being constant, not the values themselves.

**4. Over-differencing.**
If you difference a stationary series, you introduce unnecessary moving-average structure. If you difference a random walk twice, you get an MA(1) process that's harder to model than the original. Difference only until stationarity is achieved.

**5. Ignoring multiple seasonalities.**
Daily data often has both weekly and annual seasonality. A SARIMA model with s = 7 misses the annual pattern. Solutions: Fourier features for the longer seasonality, or use models that handle multiple seasonal periods (TBATS, Prophet, structural time series).

**6. Not evaluating on a meaningful baseline.**
Always compare your model to naïve baselines (seasonal naïve, last-value-carried-forward). A sophisticated model that can't beat seasonal naïve isn't adding value. The M4 and M5 competitions showed that many complex models fail to beat simple benchmarks.

**7. Overfitting deep learning models to short series.**
Transformers and LSTMs need substantial data. Fitting a 50-million-parameter model to a 200-point time series will overfit catastrophically. Use simpler methods for short series, or use global models that pool data across series.

**8. Ignoring the forecast horizon when choosing a model.**
ARIMA is excellent for short-horizon forecasts but its long-horizon forecasts revert to the mean. For long horizons, models that incorporate causal features (regressions with external drivers) often outperform pure time series models.

**9. MAPE on intermittent data.**
Intermittent demand (many zeros) makes MAPE explode or be undefined. Use MASE, RMSSE, or custom loss functions instead.

**10. Assuming stationarity after one ADF test.**
The ADF test has low power against near-unit-root alternatives. Supplement with KPSS, visual inspection, and domain knowledge. Different tests can give contradictory results — this is expected and informative.

**11. Ignoring prediction intervals.**
A point forecast without uncertainty is dangerous. Decision-makers need to know the range of possible outcomes. Always produce prediction intervals and evaluate their calibration.

**12. Treating all time series methods as competing.**
In practice, ensembles of diverse methods often outperform any single method. The M4 competition winner was an ensemble of exponential smoothing and a neural network.

---

## Part X: Interview Preparation

### 16. Interview Questions and Answers

---

#### Tier 1: Foundational Questions

**Q1: What is stationarity, and why is it important for time series modeling?**

**Answer**: A time series is stationary if its mean, variance, and autocovariance structure are constant over time. Formally, for all t and k: E[yₜ] = μ, Var(yₜ) = σ², and Cov(yₜ, yₜ₊ₖ) = γ(k) depends only on the lag k, not on t.

Stationarity matters because most classical models (AR, MA, ARMA) are derived under the stationarity assumption. Parameter estimation via MLE or OLS requires that the statistical properties of the training data are representative of the future. If the mean is trending upward during training, but the model assumes a constant mean, forecasts will be systematically wrong.

Practically, we achieve stationarity through differencing (ARIMA's "I" component removes trends), log transformations (stabilize variance), or seasonal differencing (remove seasonal patterns). We verify stationarity using ADF (tests for unit root) and KPSS (tests for stationarity) tests, noting that these can conflict — which itself is informative about the nature of non-stationarity.

**Q2: Explain the difference between AR and MA models. How do you decide which to use?**

**Answer**: An AR(p) model expresses the current value as a linear combination of p past values: yₜ = c + φ₁yₜ₋₁ + ⋯ + φₚyₜ₋ₚ + εₜ. An MA(q) model expresses the current value as a linear combination of q past errors: yₜ = μ + εₜ + θ₁εₜ₋₁ + ⋯ + θ_qεₜ₋q.

The key conceptual difference is the nature of memory. An AR process has theoretically infinite memory — a shock persists forever (though its influence decays geometrically). An MA(q) process has finite memory of exactly q steps — a shock is completely forgotten after q periods.

To decide which to use, examine the ACF and PACF of the (stationary) series. If the PACF cuts off sharply after lag p while the ACF decays gradually, use AR(p). If the ACF cuts off after lag q while the PACF decays gradually, use MA(q). If both decay gradually, an ARMA or ARIMA model is appropriate, and you select orders using information criteria (AIC, BIC).

In practice, the distinction matters less than it seems — an AR(∞) can be represented as a finite MA and vice versa (duality theorem). The practical question is which representation gives a more parsimonious (fewer parameters) description of the data.

**Q3: What is exponential smoothing, and how does it relate to ARIMA?**

**Answer**: Exponential smoothing produces forecasts that are weighted averages of past observations, with weights that decay exponentially. The simplest form, SES, uses the recursion ŷₜ₊₁ = αyₜ + (1−α)ŷₜ, where α ∈ (0, 1) controls how fast old observations are downweighted.

The family includes SES (level only), Holt's method (level + trend), and Holt-Winters (level + trend + seasonality), each adding components to handle more complex patterns.

The connection to ARIMA is precise: SES is mathematically equivalent to ARIMA(0,1,1) with θ₁ = −(1−α). More generally, every exponential smoothing method has an ARIMA equivalent. The ETS framework (Error-Trend-Seasonality) formalizes exponential smoothing as state space models with specific innovation structures, making the connection to ARIMA transparent through the state space representation.

**Q4: What is the Kalman filter, and why is it useful for time series?**

**Answer**: The Kalman filter is an algorithm for recursively estimating the state of a linear dynamical system from noisy observations. It alternates between a predict step (project the state forward using the system dynamics) and an update step (correct the prediction using the new observation).

The key quantity is the Kalman gain K, which optimally balances trust between the prediction (based on the model) and the observation (which is noisy). When observations are reliable (low observation noise R), K is large — we trust the data. When the state dynamics are uncertain (high process noise Q), K is also large — we need data to correct our uncertain predictions.

For time series, the Kalman filter is useful because: it handles missing data naturally (skip the update step), it provides proper uncertainty quantification through the state covariance P, it accommodates time-varying parameters (the state can change over time), and it unifies ARIMA and exponential smoothing under a common framework. Extensions like the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) handle nonlinear systems.

**Q5: Name five forecasting evaluation metrics and explain when each is most appropriate.**

**Answer**: 

(1) **MAE**: Best when the cost of error is proportional to magnitude and the scale is interpretable. It's symmetric and in original units. The optimal point forecast under MAE is the conditional median.

(2) **MAPE**: Best for communicating percentage accuracy to business stakeholders who want scale-free comparisons. But it's asymmetric, undefined for zero actuals, and inflates errors for small values. Avoid for intermittent or near-zero series.

(3) **MASE**: Best for rigorous accuracy benchmarking. It's scale-free, symmetric, defined for zero actuals, and provides a meaningful benchmark (MASE < 1 means you beat seasonal naïve). Recommended as the primary metric in the forecasting literature.

(4) **Quantile loss (pinball loss)**: Best for evaluating probabilistic forecasts. It assesses whether specific quantiles of the predictive distribution are calibrated. Used in settings where different types of errors have different costs (e.g., understocking is worse than overstocking).

(5) **CRPS**: Best for comprehensive evaluation of full probabilistic forecasts. It generalizes MAE to distributions, rewarding both calibration and sharpness. It's the gold standard for probabilistic forecast evaluation.

---

#### Tier 2: Mathematical Questions

**Q6: Derive the stationarity condition for an AR(1) process.**

**Answer**: Consider yₜ = φyₜ₋₁ + εₜ where εₜ ~ WN(0, σ²).

By recursive substitution: yₜ = φ(φyₜ₋₂ + εₜ₋₁) + εₜ = φ²yₜ₋₂ + φεₜ₋₁ + εₜ. Continuing:

$$
y_t = \phi^k y_{t-k} + \sum_{j=0}^{k-1} \phi^j \varepsilon_{t-j}
$$

For this to converge as k → ∞ (necessary for a well-defined stationary process), we need |φ| < 1, so that φᵏ → 0. Then:

$$
y_t = \sum_{j=0}^{\infty} \phi^j \varepsilon_{t-j}
$$

This is an infinite MA representation. The mean is E[yₜ] = 0 (constant). The variance is:

$$
\text{Var}(y_t) = \sum_{j=0}^{\infty} \phi^{2j} \sigma^2 = \frac{\sigma^2}{1 - \phi^2}
$$

which is finite and constant only when |φ| < 1. The autocovariance at lag k is γ(k) = σ²φᵏ/(1 − φ²), which depends only on k, not on t. All three stationarity conditions are satisfied if and only if |φ| < 1.

For general AR(p), we write the process using the backshift operator: (1 − φ₁B − ⋯ − φₚBᵖ)yₜ = εₜ. The characteristic equation is 1 − φ₁z − ⋯ − φₚzᵖ = 0. Stationarity requires all roots z to satisfy |z| > 1 (lie outside the unit circle).

**Q7: Derive the forecast error variance for an AR(1) model at horizon h.**

**Answer**: The h-step-ahead forecast from time T is ŷₜ₊ₕ = c(1 − φʰ)/(1 − φ) + φʰyₜ (derived via recursive substitution of the conditional expectation).

The actual value is:

$$
y_{T+h} = \phi^h y_T + c\frac{1-\phi^h}{1-\phi} + \sum_{j=0}^{h-1} \phi^j \varepsilon_{T+h-j}
$$

The forecast error is:

$$
e_{T+h|T} = y_{T+h} - \hat{y}_{T+h|T} = \sum_{j=0}^{h-1} \phi^j \varepsilon_{T+h-j}
$$

since the terms involving yₜ and c cancel. The variance is:

$$
\text{Var}(e_{T+h|T}) = \sum_{j=0}^{h-1} \phi^{2j} \sigma^2 = \sigma^2 \frac{1 - \phi^{2h}}{1 - \phi^2}
$$

As h → ∞, this converges to σ²/(1 − φ²), the unconditional variance of the process. This makes intuitive sense: for very long horizons, our forecast is just the unconditional mean, so the forecast error variance equals the variance of the process itself.

A 95% prediction interval at horizon h is:

$$
\hat{y}_{T+h} \pm 1.96 \cdot \sigma \sqrt{\frac{1 - \phi^{2h}}{1 - \phi^2}}
$$

Key insight: the prediction interval widens with h but is bounded. This contrasts with an ARIMA(0,1,0) (random walk) where the prediction interval grows without bound: Var(eₕ) = hσ².

**Q8: How does the LSTM avoid the vanishing gradient problem? Be mathematically specific.**

**Answer**: In a vanilla RNN, the gradient of the loss with respect to early hidden states involves a product of Jacobians:

$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=t+1}^{T} \text{diag}(\tanh'(z_k)) \cdot W_{hh}
$$

where zₖ is the pre-activation. Since |tanh'(z)| ≤ 1 and ‖Wₕₕ‖ is typically < 1 after initialization, this product shrinks exponentially with T − t.

In an LSTM, the key gradient path goes through the cell state:

$$
\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^{T} \frac{\partial c_k}{\partial c_{k-1}} = \prod_{k=t+1}^{T} f_k
$$

where fₖ = σ(Wf[hₖ₋₁, xₖ] + bf) is the forget gate at step k. Crucially, this is an element-wise product of sigmoid outputs (values in [0, 1]), not a matrix product. When the forget gate is close to 1 (the default bias is often initialized to 1 or higher to encourage this), the gradient passes through unchanged: ∂cₜ/∂cₜ₋₁ ≈ I.

There's no Jacobian of tanh in this path, and no weight matrix multiplication. The cell state acts as a "gradient highway." The network can learn to set the forget gate near 1 for information it wants to remember, maintaining gradient flow across hundreds of steps.

Note: this doesn't mean LSTM gradients *never* vanish — they can still vanish through the hidden state path hₜ. But the cell state path provides at least one well-behaved gradient path, which is sufficient for learning.

**Q9: Explain ProbSparse attention in the Informer and derive its complexity.**

**Answer**: Standard self-attention computes A = softmax(QKᵀ/√d)V with Q, K, V ∈ ℝᵀˣᵈ. Computing QKᵀ costs O(T²d), which is prohibitive for long sequences (T = 10,000+).

Informer's key observation: in practice, most attention distributions are close to uniform — only a few queries have "sharp" attention patterns (concentrating on specific keys). Queries with near-uniform attention contribute little information; they can be approximated.

To identify "important" queries, Informer defines the KL divergence between query qᵢ's attention distribution and the uniform distribution:

$$
M(q_i, K) = \ln \sum_{j} e^{q_i k_j^\top / \sqrt{d}} - \frac{1}{T}\sum_j \frac{q_i k_j^\top}{\sqrt{d}}
$$

A high M means the query has a peaked attention distribution (it's "active"). A low M means near-uniform attention ("lazy" query).

Informer approximates M by sampling a subset of keys (size c·ln T) and computing M on the subset. It then selects the top u = c·T·ln T queries with the highest M values and computes full attention only for those. Remaining queries are assigned the mean value of V.

The complexity becomes O(T·ln T · d) for computing the sampled M values, plus O(u·T·d) = O(c·T·ln T · T · d) for the selected queries... but in practice, u << T, and the distilling mechanism further reduces sequence length at each layer, so the overall complexity is O(T log T).

**Q10: Derive the optimal reconciliation formula for hierarchical forecasting.**

**Answer**: Let ŷ ∈ ℝᵐ be the vector of independent base forecasts at all m levels. Let S ∈ ℝᵐˣⁿ be the summing matrix that maps the n bottom-level series to all m levels. Any coherent forecast has the form Sb for some bottom-level forecast b ∈ ℝⁿ.

We want to find b that minimizes the weighted distance to the base forecasts:

$$
\min_b (\hat{y} - Sb)^\top W^{-1} (\hat{y} - Sb)
$$

where W is a positive definite weight matrix (e.g., the covariance of base forecast errors).

Taking the derivative and setting to zero:

$$
-2S^\top W^{-1}(\hat{y} - Sb) = 0
$$

$$
S^\top W^{-1} S b = S^\top W^{-1} \hat{y}
$$

$$
b = (S^\top W^{-1} S)^{-1} S^\top W^{-1} \hat{y}
$$

The reconciled forecast at all levels is:

$$
\tilde{y} = Sb = S(S^\top W^{-1} S)^{-1} S^\top W^{-1} \hat{y}
$$

Common choices for W:

- W = I: OLS reconciliation. Simple but ignores accuracy differences across levels.
- W = diag(Ŵ₁): Weighted least squares. Uses diagonal of forecast error covariance (accounts for different accuracies but ignores correlations).
- W = Ŵ₁ (full covariance): MinT (Minimum Trace) reconciliation. Optimal but requires estimating the full covariance matrix, which is high-dimensional.
- W = diag(Ŝ): Scaling by in-sample error variance. A practical compromise.

The reconciled forecasts are guaranteed to be coherent (satisfy the aggregation constraints) and, under the MinT approach, have minimum variance among all coherent forecasts that are unbiased.

---

#### Tier 3: Applied Questions

**Q11: You're tasked with building a demand forecasting system for an e-commerce company with 1 million products. Walk through your approach.**

**Answer**: 

**Data exploration**: Start by segmenting products by demand pattern. Use the ADI (Average Demand Interval) and CV² (squared coefficient of variation) framework to classify series as smooth, erratic, lumpy, or intermittent. Each class may require different modeling approaches. Intermittent demand (many zeros) needs specialized methods like Croston's method or IMAPA.

**Feature engineering**: Create a tabular dataset pooled across all products. Features include lag features (lags matching the forecast horizon), rolling statistics at multiple windows, calendar features (day-of-week, holidays, seasonal Fourier terms), product metadata (category, price tier, age), promotional indicators, and any available external features (weather, macroeconomic indicators).

**Model selection**: Use a global LightGBM model as the primary approach, for several reasons: it handles millions of training rows efficiently, captures nonlinear interactions between features, and pools statistical strength across products. Train a single model on the pooled data with product category embeddings or one-hot features.

For the top 100 highest-revenue products, also fit individual SARIMA or ETS models as a sanity check and potential ensemble member.

**Evaluation**: Use a temporal train/test split with walk-forward validation. Primary metric: WMAPE (weighted MAPE gives more importance to high-volume products). Secondary metrics: MASE (for benchmark comparison) and quantile coverage (for prediction intervals). Compare against a seasonal naïve baseline.

**Production considerations**: Set up automated retraining on a weekly schedule. Monitor WMAPE over time and per product category. Implement change point detection to flag products with suddenly changing demand patterns. Handle cold-start (new products) through category-level forecasts or similar-product matching.

**Hierarchical reconciliation**: If forecasts are needed at multiple levels (product, category, department), use optimal reconciliation (MinT) to ensure coherence.

**Q12: Your time series model performed well in backtesting but poorly in production. What could have gone wrong?**

**Answer**: I'd investigate several categories of failure:

**Data leakage in backtesting**: This is the most common cause. Check if any features used information that wouldn't be available at prediction time. Common culprits: lag features that use too-recent lags relative to the forecast horizon, rolling statistics that include the target observation, features derived from the full dataset (e.g., global normalization using test-period statistics). The fix is to re-examine the feature pipeline and ensure strict temporal ordering.

**Distribution shift (concept drift)**: The relationship between features and target may have changed. Seasonality patterns might have shifted (e.g., due to COVID), a new competitor entered the market, or pricing strategy changed. Diagnose by comparing feature distributions and model residuals between the backtest period and the production period. Solutions: more recent training data, shorter retraining cycles, online learning, or explicitly modeling structural breaks.

**Backtesting methodology errors**: If the backtest used a single train/test split, it might have been unrepresentatively favorable. Use walk-forward validation with multiple cutoff dates. Also check if the backtest used the same prediction infrastructure as production (data pipelines, feature computation, etc.) — discrepancies between offline and online environments are a common source of degradation.

**Data quality issues in production**: Missing values, delayed data feeds, changed data schemas, or changed feature definitions. The training data might have been clean, but production data often isn't. Add data quality monitoring and alerting.

**Temporal aggregation mismatch**: Training on daily data to produce weekly forecasts (aggregating daily predictions) can introduce errors differently than training directly on weekly data. Ensure the forecast frequency matches the decision-making frequency.

**Outlier/event handling**: The production period might contain events (holidays, promotions, outages) that weren't in the training data or weren't properly handled by the model. Ensure the model has appropriate event features and has been evaluated on periods containing similar events.

**Q13: When would you choose a Transformer-based model (TFT, Informer) over ARIMA or XGBoost for time series forecasting?**

**Answer**: Transformer-based models are most appropriate when several conditions are met simultaneously:

**Data volume**: Transformers are data-hungry. They shine when you have hundreds or thousands of related time series (enabling global training) or very long individual series (thousands of observations). For a single series with 100 observations, ARIMA or ETS will outperform any Transformer.

**Complex dependencies**: When there are long-range, nonlinear dependencies — for example, in energy forecasting where weather 2 weeks ago affects building temperature today through thermal inertia, or in retail where promotional calendars create complex interaction effects. ARIMA handles only linear dependencies, and XGBoost with lag features has a fixed lookback window.

**Rich input features**: TFT specifically excels when you have a mix of static covariates (e.g., product category), known future inputs (e.g., planned promotions, calendar), and past-observed inputs (e.g., past sales). Its variable selection networks learn which inputs matter. XGBoost can also handle mixed features, but TFT's attention mechanism over these features is more expressive.

**Interpretability needs**: Surprisingly, TFT provides interpretability through attention weights (which historical time steps does the model focus on?) and variable importance scores. This is more interpretable than an LSTM, though less interpretable than ARIMA coefficients.

**When NOT to use Transformers**: Short individual series, limited data, real-time latency requirements (Transformers are compute-heavy), simple seasonal patterns easily captured by ETS, or when a simpler model already achieves near-optimal performance. In the M4 forecasting competition (100,000 individual series), pure statistical methods outperformed early deep learning approaches — scale and data volume are prerequisites for deep learning success.

**Q14: How would you design an anomaly detection system for a streaming time series with evolving seasonal patterns?**

**Answer**: I'd build a multi-layer detection system:

**Layer 1 — Adaptive decomposition**: Use STL with a rolling window to decompose the series into trend, seasonal, and remainder. The rolling window ensures the seasonal component can evolve. Alternative: a structural time series model with time-varying seasonality estimated via the Kalman filter.

**Layer 2 — Statistical thresholding on residuals**: After removing trend and seasonality, compute a rolling robust estimate of the residual distribution (e.g., rolling median and MAD — Median Absolute Deviation). Flag points where the residual exceeds k × MAD (e.g., k = 3). Using MAD instead of standard deviation makes this robust to the anomalies themselves contaminating the threshold.

**Layer 3 — Change point detection**: Run BOCPD (Bayesian Online Change Point Detection) on the residuals to detect persistent shifts in mean or variance. This distinguishes a one-off spike (point anomaly, Layer 2 catches it) from a lasting regime change (requires model update).

**Handling evolving seasonality**: Key challenge. If the seasonal pattern changes (e.g., a store's traffic pattern shifts because of a new competitor), the decomposition must adapt. Solutions include: a short seasonal window in STL (but this increases noise), explicit modeling of seasonal change points, or a Kalman filter formulation where the seasonal states have non-zero process noise (allowing gradual drift).

**False positive management**: Set thresholds using a validation period with known anomalies (or expert labels). Use a two-stage system: automated detection flags candidates, and a human review step confirms. Track precision and recall over time. Implement a feedback loop where confirmed/rejected anomalies are used to calibrate thresholds.

**Streaming considerations**: The entire pipeline must operate in O(1) memory per time step (constant memory, not growing with series length). Kalman filter and BOCPD naturally support this. STL can be adapted by maintaining a fixed-size buffer.

---

#### Tier 4: Debugging & Failure Mode Questions

**Q15: Your ARIMA model's residuals show significant autocorrelation at lag 7. What does this mean and how do you fix it?**

**Answer**: Significant autocorrelation at lag 7 means the model is systematically failing to capture weekly seasonality. The residuals contain predictable weekly patterns — information the model should have captured but didn't.

Diagnosis: This likely means either the seasonal differencing was omitted or the seasonal order was set incorrectly. If you're using ARIMA(p,d,q) without seasonal components on daily data, weekly patterns will leak into the residuals.

Fix: Switch to SARIMA(p,d,q)(P,D,Q)₇ with s = 7. Start with seasonal differencing (D = 1) and examine the ACF/PACF at seasonal lags (7, 14, 21, ...). If the seasonal ACF cuts off after lag 7, try Q = 1. If the seasonal PACF cuts off after lag 7, try P = 1.

Alternative: If SARIMA with s = 7 is insufficient (maybe there's also annual seasonality at s = 365.25, which SARIMA can't handle), switch to a regression with ARIMA errors, including Fourier terms for the weekly cycle: sin(2πt/7), cos(2πt/7), sin(4πt/7), cos(4πt/7), etc.

Verify the fix by re-examining the Ljung-Box test on the new model's residuals at both non-seasonal and seasonal lags.

**Q16: Your global LightGBM forecasting model works well overall but produces terrible forecasts for new products with less than 4 weeks of history. What's happening?**

**Answer**: The model relies heavily on lag features and rolling statistics, which are mostly missing or unreliable for products with < 4 weeks of history. With lag_7 and lag_14 features being null or based on very few observations, the model has almost no signal to work with.

Solutions, ordered by implementation effort:

(1) **Feature imputation from similar products**: For new products, find the closest existing products by metadata (same category, similar price, same brand) and use their historical features as a proxy. This is essentially a nearest-neighbor cold-start strategy.

(2) **Separate cold-start model**: Train a model specifically for new products that uses only metadata features (category, price, season of launch) and aggregate category-level demand patterns. After 4 weeks, transition to the main model.

(3) **Hierarchical approach**: Use category-level or brand-level forecasts (which have long history) and disaggregate using metadata-based proportions. As the new product accumulates data, gradually shift weight from the disaggregated forecast to the product-level forecast.

(4) **Bayesian/prior approach**: Use DeepAR or a similar model that can learn a prior distribution over time series from the training data. New products start with this prior and are updated as data arrives.

(5) **Feature engineering**: Create features that are always available regardless of history length: calendar features (day of week, holiday), metadata features (category, price), and global features (overall site traffic, category-level demand). Ensure the model can function using only these features.

**Q17: You trained an LSTM for stock price prediction. It achieves near-perfect accuracy on the training set and very poor accuracy on the test set. The test set is the most recent 20% of data. What went wrong?**

**Answer**: Several things are likely at play:

**Overfitting**: An LSTM with sufficient capacity will memorize the training series, achieving near-zero training error. This is especially likely with a single univariate series. LSTMs have thousands or millions of parameters but a single stock price series might have only ~5000 daily observations. The model memorizes noise rather than learning generalizable patterns.

**The efficient market hypothesis problem**: Stock prices are widely believed to follow (approximately) a random walk. If the best predictor of tomorrow's price is today's price, then any model that appears to do better on training data is fitting noise. The "near-perfect training accuracy" might actually be the LSTM learning to output yₜ ≈ yₜ₋₁ (the last observed value) plus some noise patterns specific to the training period.

**Look-ahead bias check**: Verify that the training data doesn't inadvertently contain information from the future — this is surprisingly common in financial datasets (survivorship bias, adjusted prices, etc.).

**Scale of "accuracy"**: What metric is being used? If it's R² on price levels, even a random walk achieves high R² because prices are autocorrelated. Use metrics on *returns* (Δyₜ = yₜ − yₜ₋₁), not levels. An RMSE of $1 on a $500 stock looks great but might be worse than naïve.

**Regime change**: The test period (most recent 20%) might correspond to a different market regime (e.g., the training period was a bull market, the test period is volatile). The model learned patterns specific to one regime.

Fixes: Regularize heavily (dropout, weight decay), use walk-forward validation instead of a single split, simplify the model (maybe a simple linear model or AR is more appropriate), augment with fundamental features rather than relying on price alone, and most importantly — lower expectations. Consistent alpha generation from price prediction is extraordinarily hard.

**Q18: Your forecasting pipeline uses rolling 28-day lag features and 7-day rolling mean features. Users report that forecasts are stale — they don't react to a sudden demand surge that started 3 days ago. Why?**

**Answer**: The problem is the gap between the most recent available data and the forecast target.

If the forecast horizon is h = 7 days ahead, the most recent lag the model can use without leakage is lag_7 (the value 7 days ago). The 28-day lag looks 28 days back. The 7-day rolling mean uses data from days t−7 through t−13 (to avoid leakage). None of these features capture what happened in the last 3 days.

So the demand surge that started 3 days ago is completely invisible to the model at forecast time. The model won't "see" the surge until 4 more days pass (when the surge enters the lag_7 feature).

Solutions:

(1) **Reduce the forecast horizon** if the business process allows. If you forecast 1 day ahead instead of 7, lag_1 is available, and the model reacts within 1 day.

(2) **Use a separate "recent trend" feature** that captures the direction of very recent changes: e.g., the slope of a linear fit to the last 3 known observations, or the ratio of the last known observation to the rolling mean. Even though we can't use lag_1 for a 7-day forecast, we can compute features from the known data up to 7 days ago that capture whether demand was trending up or down.

(3) **Multi-step update strategy**: Generate a base forecast at the beginning of the week, but update it daily using a Bayesian update or a correction model. As each day's actual demand comes in, adjust the remaining days' forecasts.

(4) **Event detection**: Build a separate detection system that identifies sudden demand surges (change point detection) and triggers an immediate forecast override or recalibration.

---

#### Tier 5: Follow-Up and Probing Questions

**Q19: You mentioned MASE uses the seasonal naïve as a baseline. What if your data has multiple seasonalities? Which one do you use?**

**Answer**: Standard MASE uses a single seasonal period m for the naïve denominator. When multiple seasonalities exist (e.g., weekly m = 7 and annual m = 365.25 for daily data), the choice matters:

The most principled approach is to use the dominant seasonality — the one that explains the most variance. For daily retail data, weekly patterns are usually stronger than annual patterns at the daily level, so m = 7 is typical.

However, you could also compute MASE relative to a multi-seasonal naïve baseline. For example, the STL-naïve: decompose using STL with multiple seasonal components, and use the full decomposition (trend + all seasonal components) as the naïve forecast, with only the residual variance as the scaling denominator.

Alternatively, some practitioners report multiple MASE values (one for each seasonal period) to understand at which temporal scale the model adds value.

The deeper insight is that MASE is a relative metric, and the choice of baseline defines what "good" means. If your application's decision frequency aligns more with weekly patterns, use m = 7. If annual planning is the goal, m = 365 might be more relevant. Always be explicit about which baseline you're using.

**Q20: You said Transformers have O(T²) complexity. But TCNs can also model very long sequences. When would you prefer one over the other?**

**Answer**: The choice depends on several factors:

**TCN advantages**: Faster training (fully parallel convolutions vs. attention's quadratic cost), more memory-efficient for long sequences, stable gradient flow through residual connections, and a fixed, well-understood receptive field. TCNs are particularly good when you know the relevant history length in advance — you can set the architecture's receptive field to exactly that length.

**Transformer advantages**: Adaptive attention patterns — the model learns which historical time steps to focus on, which can vary per prediction. A TCN with receptive field R treats all positions within R equally (the learned convolution weights are shared across time), while attention explicitly selects which positions matter. Transformers are better when the relevant history length varies (e.g., some predictions need last week's data, others need same-day-last-year).

**Practical heuristic**: For problems with clear, fixed-length dependencies (e.g., hourly energy forecasting with 24-hour and 168-hour cycles), TCNs are often more efficient and equally effective. For problems with variable-length, complex dependencies (e.g., retail demand influenced by irregular promotions, competitor actions, and events at various horizons), Transformers' flexible attention is more valuable.

Efficiency-minded architectures like Informer (O(T log T) attention) narrow the computational gap, but TCNs remain faster in practice for most problems.

**Q21: In hierarchical forecasting, you mentioned MinT reconciliation. What happens if the covariance matrix W is singular or poorly estimated?**

**Answer**: This is a real practical problem. The covariance matrix W of base forecast errors is m × m, where m is the total number of series across all levels. For large hierarchies (e.g., 50,000 products across 500 stores at 3 hierarchy levels), m can be enormous, and estimating a full covariance matrix from limited forecast errors is infeasible.

**Singularity**: With k evaluation periods and m > k series, the sample covariance matrix is rank-deficient (singular). Even with m < k, near-singularity causes numerical instability.

**Practical solutions**:

(1) **Diagonal W (WLS reconciliation)**: Only estimate variances, ignoring correlations. W = diag(σ₁², ..., σₘ²). This is always invertible and works well in practice, often nearly matching full MinT.

(2) **Shrinkage estimation**: Use Ledoit-Wolf shrinkage to estimate W: Ŵ = (1−λ)S + λ·diag(S), where S is the sample covariance and λ ∈ [0, 1] is the shrinkage intensity. This regularizes the covariance toward a diagonal structure, ensuring invertibility.

(3) **Structural assumptions**: The HTS (Hierarchical Time Series) literature shows that for certain hierarchy structures, W can be parameterized with far fewer parameters. For example, assuming errors are uncorrelated across bottom-level series simplifies W dramatically.

(4) **Monte Carlo reconciliation**: Instead of an analytical formula requiring W⁻¹, use simulation-based approaches that draw from the base forecast distributions and project onto the coherent subspace.

The key practical advice: start with diagonal W (WLS). It's simple, stable, and performs surprisingly well. Only move to shrinkage-based MinT if you have enough forecast errors to estimate correlations reliably.

**Q22: How does the Kalman filter relate to Bayesian inference?**

**Answer**: The Kalman filter is exact Bayesian inference for the linear Gaussian state space model.

At each time step, we have a prior on the state: p(xₜ | y₁:ₜ₋₁) = N(x̂ₜ|ₜ₋₁, Pₜ|ₜ₋₁). This prior comes from the predict step (propagating the previous posterior through the state dynamics).

We observe yₜ, which gives us a likelihood: p(yₜ | xₜ) = N(Hₜxₜ, Rₜ).

Bayes' theorem gives the posterior: p(xₜ | y₁:ₜ) ∝ p(yₜ | xₜ) p(xₜ | y₁:ₜ₋₁). Since both prior and likelihood are Gaussian, the posterior is also Gaussian: p(xₜ | y₁:ₜ) = N(x̂ₜ|ₜ, Pₜ|ₜ). The update equations of the Kalman filter compute the mean and covariance of this posterior analytically.

The Kalman gain Kₜ = Pₜ|ₜ₋₁Hₜᵀ(HₜPₜ|ₜ₋₁Hₜᵀ + Rₜ)⁻¹ is the Bayesian optimal weighting between prior and likelihood.

This connection is important because it generalizes. For nonlinear state space models, exact Bayesian inference is intractable, but we can use:
- **Extended Kalman Filter (EKF)**: Linearize the system and apply standard Kalman. Fast but can be inaccurate for highly nonlinear systems.
- **Unscented Kalman Filter (UKF)**: Propagate a set of "sigma points" through the nonlinear functions. Better accuracy than EKF, still deterministic.
- **Particle Filter**: Monte Carlo approximation of the posterior. Handles arbitrary nonlinearity and non-Gaussianity but is computationally expensive (O(N) where N is the number of particles).

For practitioners: if your state space model is linear and Gaussian, use the Kalman filter — it's exact and efficient. If nonlinear but mildly so, try UKF. If highly nonlinear or non-Gaussian, use particle filters or variational methods.

**Q23: Your forecasting system uses an ensemble of ARIMA, LightGBM, and an LSTM. How should you combine the forecasts?**

**Answer**: Forecast combination is a well-studied problem with several approaches:

**Simple average**: Surprisingly powerful. Decades of research (the "forecast combination puzzle") show that simple averaging of diverse forecasts often outperforms more sophisticated combination methods. The key is diversity — the models should make different types of errors.

**Inverse-variance weighting**: Weight each model inversely proportional to its validation error variance: wᵢ = (1/σᵢ²) / Σⱼ(1/σⱼ²). More accurate models get more weight. This is optimal when errors are uncorrelated, which is unlikely but serves as a reasonable starting point.

**Stacking (meta-learning)**: Train a meta-model that takes the three base forecasts as inputs and learns the optimal combination weights. Use time-series-appropriate cross-validation (walk-forward) to avoid overfitting the meta-model. The meta-model can learn nonlinear combinations and time-varying weights. A linear meta-model with non-negative constrained weights is a good starting point.

**Bayesian Model Averaging**: Assign posterior model probabilities based on out-of-sample likelihood and combine forecasts weighted by these probabilities. Provides proper uncertainty quantification for the combined forecast.

**Dynamic combination**: Different models may be best in different regimes (ARIMA in stable periods, LSTM during complex pattern changes). Use a regime-switching framework or a meta-model with regime features.

**Practical recommendation**: Start with a simple average. Measure the improvement from moving to stacking with walk-forward validation. Often, the improvement over simple averaging is small, and the added complexity of the meta-model isn't worth it. However, if the base models have very different accuracy levels, inverse-variance weighting or stacking can meaningfully improve.

**Q24: Explain how you would handle a time series with irregular (non-uniform) time stamps.**

**Answer**: Irregular time series are common in event-driven systems (web clickstreams, sensor data with adaptive sampling, medical records). Most time series methods assume uniform spacing.

**Approach 1 — Resample to regular grid**: Aggregate or interpolate to create uniform intervals. For aggregation, sum or average values within each bin (e.g., resample from variable event times to hourly counts). For interpolation (when you need values at specific times), use linear interpolation, spline interpolation, or carry-forward/backward.

Trade-off: Resampling introduces information loss. The choice of interval matters — too coarse loses signal, too fine creates many empty/imputed values.

**Approach 2 — Time-aware models**: Use models that natively handle irregular timing. The Kalman filter handles irregular observations naturally (just apply the predict step with the actual time gap Δt, and update whenever an observation arrives). Neural ODEs and continuous-time models (like Neural Controlled Differential Equations) parameterize the dynamics in continuous time and can handle any observation schedule.

**Approach 3 — Time-as-feature**: Include the time gap since the last observation as a feature. This lets standard models (LightGBM, LSTM) learn that a value after a long gap might have different significance than after a short gap. For LSTMs, the "time-aware LSTM" modifies the forget gate to incorporate time gaps: fₜ = σ(Wf[hₜ₋₁, xₜ, Δt] + bf).

**Approach 4 — Point processes**: If the irregular timing itself carries information (e.g., bursts of activity), model the event times using a temporal point process (Hawkes process, neural Hawkes). This jointly models when events occur and their marks (values).

Practical recommendation: If the irregularity is mild and the time series has a natural frequency (e.g., sensor data that's "mostly" every 5 minutes with some gaps), resampling is simplest. If the irregularity is fundamental to the data-generating process, use time-aware models.
