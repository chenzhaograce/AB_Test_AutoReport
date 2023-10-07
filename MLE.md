# Concept of MLE

## MLE usage

MLE stands for "Maximum Likelihood Estimation," and it's a statistical concept used to find the best-fitting model or parameters for a given dataset. Let me explain it in simple words:

Imagine you have a set of data points, and you want to figure out the most likely way a certain process generated those data points. MLE helps you find the values for the process's parameters that make the observed data the most probable or likely.

Here's a common example:

**Coin Tossing:** Suppose you're flipping a coin, and you want to know the probability of getting heads. You flip the coin 100 times and get 60 heads and 40 tails. You suspect the coin might be biased.

MLE helps you estimate the bias of the coin (the probability of getting heads) that best explains your results. In this case, MLE might tell you that the coin has a 0.6 probability of landing heads up, as it maximizes the likelihood of getting 60 heads out of 100 tosses.

Typical Usages of MLE:

1. **Statistical Modeling:** MLE is used in various statistical models, like linear regression, logistic regression, and many others. It helps find the best parameters that fit the observed data.

2. **Machine Learning:** In machine learning, MLE is often used in training models to find the optimal parameters (weights and biases) that minimize the difference between model predictions and actual data.

3. **Probability Distributions:** MLE is used to estimate parameters in probability distributions (e.g., Gaussian distribution, Poisson distribution) based on observed data.

4. **Natural Language Processing:** In NLP, MLE can be used to estimate probabilities for words or phrases in language models, helping machines generate more human-like text.

In essence, MLE is about finding the most likely explanation or settings for a model given the data you have. It's a fundamental concept in statistics and machine learning that helps make predictions and decisions based on data.

## MLE in mathematics

Maximum Likelihood Estimation (MLE) is a method for finding the most likely values of the parameters of a statistical model given some observed data. Let's walk through the MLE process step by step using a simple example of estimating the mean of a Gaussian (normal) distribution.

**Step 1: Define the Likelihood Function**

In MLE, we start by defining the likelihood function. The likelihood function tells us how likely the observed data is for different values of the model's parameters. For a Gaussian distribution with an unknown mean (μ) and a known standard deviation (σ), the likelihood function is:

$L(\mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x_i - \mu)^2}{2\sigma^2}}$

Here:
- $L(\mu)$ is the likelihood function.
- $\mu$ is the parameter we want to estimate (the mean).
- $x_i$ are the observed data points.
- $n$ is the number of data points.
- $\sigma$ is the known standard deviation.

**Step 2: Take the Logarithm**

It's often more convenient to work with the logarithm of the likelihood function, called the log-likelihood. Taking the natural logarithm (ln) simplifies calculations:

$ln(L(\mu)) = \sum_{i=1}^{n} \left( -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2} \right)$

**Step 3: Find the Derivative**

To find the maximum likelihood estimate for \(\mu\), we need to find the value of \(\mu\) that maximizes the log-likelihood. To do this, we take the derivative of the log-likelihood with respect to \(\mu\) and set it equal to zero:

$\frac{d}{d\mu}ln(L(\mu)) = \sum_{i=1}^{n} \frac{x_i - \mu}{\sigma^2} = 0$

**Step 4: Solve for the Maximum Likelihood Estimate**

Now, we solve for $\mu$ by rearranging the equation:

$\sum_{i=1}^{n} (x_i - \mu) = 0$

$\sum_{i=1}^{n} x_i - n\mu = 0$

$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$

So, the maximum likelihood estimate for $\mu$ is simply the sample mean of the observed data points.

In summary, MLE is a method for finding the parameter values that maximize the likelihood of observing the given data. In this example, we found that the MLE for the mean ($\mu$) of a Gaussian distribution is the sample mean of the observed data. The process involves defining the likelihood function, taking its logarithm, finding the derivative, and solving for the parameter of interest. This methodology can be applied to various statistical models with different likelihood functions.