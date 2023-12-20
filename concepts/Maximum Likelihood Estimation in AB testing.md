# Maximum Likelihood Estimation in AB testing

For quite some time, we’ve employed A/B testing as a means to examine hypotheses and improve outcomes through the iteration of various experiments. Thanks to advanced tools available on A/B testing websites, we can now readily obtain sample sizes, test durations, and p-values. But how does Maximum Likelihood Estimation factor into this? In the following sections, I’ll provide a brief overview of MLE and its relevance to A/B testing.

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

## Relationship with AB testing

In summary, MLE is a method for finding the parameter values that maximize the likelihood of observing the given data. In this example, we found that the MLE for the mean ($\mu$) of a Gaussian distribution is the sample mean of the observed data. The process involves defining the likelihood function, taking its logarithm, finding the derivative, and solving for the parameter of interest. This methodology can be applied to various statistical models with different likelihood functions.

Maximum Likelihood Estimation (MLE) and A/B testing are two different but related concepts in the field of statistics and experimentation, often used together in the context of hypothesis testing and data analysis. Here's how they are related:

1. **Hypothesis Testing:** Both MLE and A/B testing involve hypothesis testing.

   - **MLE**: In MLE, you are typically trying to estimate the parameters of a statistical model, such as the mean or variance. The hypothesis testing aspect comes into play when you want to test whether a specific value of the parameter (e.g., a null hypothesis) is a good fit for your data or not.

   - **A/B Testing**: A/B testing, also known as split testing, is a method used to compare two or more versions of something (e.g., a web page, an app feature) to determine which one performs better. It involves hypotheses like the null hypothesis (no difference between versions) and the alternative hypothesis (there is a significant difference).

2. **Parameter Estimation:** MLE is often used for parameter estimation in statistical models, which can be crucial in A/B testing.

   - **MLE**: As explained earlier, MLE helps you find the parameter values that maximize the likelihood of observing your data. In A/B testing, this could be used to estimate parameters like the conversion rate, click-through rate, or other metrics for each version of the test.

   - **A/B Testing**: After estimating parameters using MLE or other methods, you can use statistical tests to compare the performance of different versions in an A/B test. For example, you might use MLE to estimate conversion rates for two web page variants and then use hypothesis testing to determine if the difference between them is statistically significant.

3. **Decision Making:** Both MLE and A/B testing ultimately play a role in decision-making based on data.

   - **MLE**: MLE helps you make decisions about the best-fitting parameters for your data distribution. For example, it might help you determine the best estimate for a conversion rate.

   - **A/B Testing**: A/B testing helps you make decisions about which version of a product or feature is more effective based on observed data. You decide whether to adopt one version over the other(s) based on statistical significance and practical significance.

In summary, MLE is a statistical technique used for parameter estimation, which can be applied within the context of A/B testing. A/B testing, on the other hand, involves hypothesis testing to compare different versions of something and make data-driven decisions. MLE can be used to estimate the parameters needed for hypothesis testing in A/B testing scenarios, helping you assess the performance of different variants accurately.

## Example of AB testing on CTR 
In the context of an A/B test where you are comparing the Click-Through Rate (CTR) of two variants (A and B), you can use Maximum Likelihood Estimation (MLE) to estimate the CTR for each variant, and then calculate the sample mean of these estimates to determine the overall test result.

Here's how you can do it step by step:

1. **Estimate CTR for each variant using MLE:** For each variant (A and B), you can use MLE to estimate the Click-Through Rate. This involves counting the number of clicks divided by the number of impressions for each variant. MLE will give you the parameter estimates that maximize the likelihood of observing the data for each variant.

2. **Calculate Sample Means:** Once you have estimated the CTR for each variant, you can calculate the sample mean for both A and B. These sample means represent the average CTR for each variant based on your sample data.

3. **Compare the Means:** To determine the overall test result, you can compare the sample means of the CTR for variants A and B. You might use hypothesis testing techniques (e.g., a t-test or z-test) to assess whether the difference in sample means is statistically significant. This will help you decide if there is a meaningful difference in CTR between the two variants.

In summary, by estimating the CTR for each variant using MLE and then calculating the sample means, you can assess the performance of the A/B test and determine if one variant has a significantly higher or lower CTR compared to the other. MLE helps you estimate the underlying parameters (CTR in this case) based on your observed data, and hypothesis testing helps you make statistical inferences about the results of your A/B test.