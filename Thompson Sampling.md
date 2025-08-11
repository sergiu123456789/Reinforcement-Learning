# Predicting the Best Ad with Thompson Sampling

This example demonstrates how to use **Thompson Sampling** to solve a reinforcement learning problem where we have:
- **10 ads**
- **10,000 users**
- Each user clicks only on certain ads.

Our goal: **Predict which ad is most likely to be clicked after all rounds.**

---

## Problem Setup

We have a dataset where each row corresponds to a user and each column corresponds to an ad.  
The value is `1` if the user clicked the ad, `0` otherwise.

We will simulate the scenario and use **Thompson Sampling** to decide which ad to show to each user.

---

## Thompson Sampling Algorithm

Thompson Sampling is a Bayesian approach for the **Multi-Armed Bandit** problem.  
At each round:
1. For each ad, we keep track of:
   - `number_of_rewards_1` (clicks)
   - `number_of_rewards_0` (no clicks)
2. We draw a random value from the Beta distribution:
   - `Beta(number_of_rewards_1 + 1, number_of_rewards_0 + 1)`
3. We select the ad with the highest sampled value.
4. We update the counts based on whether the user clicked.

This balances **exploration** (trying different ads) and **exploitation** (showing the best ad so far).

---

## Python Implementation

```python
import random
import numpy as np

# Simulate the dataset
N = 10000  # users
d = 10     # ads
ads_true_ctr = np.random.rand(d) * 0.15 + 0.05  # true click rates between 5% and 20%
dataset = np.array([[1 if random.random() < ads_true_ctr[j] else 0 for j in range(d)] for i in range(N)])

# Thompson Sampling
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
ads_selected = []

for n in range(N):
    ad = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1

# Predict the best ad
best_ad = np.argmax(numbers_of_rewards_1)
print(f"The most likely ad to be chosen is Ad #{best_ad} with estimated CTR {numbers_of_rewards_1[best_ad] / (numbers_of_rewards_1[best_ad] + numbers_of_rewards_0[best_ad]):.2%}")
```

---

## Example Output

```
The most likely ad to be chosen is Ad #3 with estimated CTR 18.50%
```

---

## Why Thompson Sampling Works Well

- **Exploration vs. Exploitation Trade-off:** The Beta distribution naturally encourages exploration when uncertainty is high and exploitation when confident.
- **Adaptability:** Works even if the underlying probabilities change over time.
- **Performance:** Often outperforms ε-greedy methods in real-world ad selection problems.

---

## Conclusion

Using Thompson Sampling, we can dynamically learn which ad is most likely to be clicked without showing all ads equally often.  
This leads to higher click-through rates and better user engagement.
