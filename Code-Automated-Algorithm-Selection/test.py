# import numpy as np
# from scipy.stats import wilcoxon
#
# # Scores of the classmates
# scores = np.array([76, 96, 74, 88, 79, 95, 75, 82, 90, 60, 77, 56])
#
# # Hypothetical median value to test against, e.g., 84
# median_value = 84
#
# # Calculate the differences from the median
# differences = scores - median_value
#
# # Apply the Wilcoxon signed-rank test
# # zero_method='wilcox' excludes zero-difference pairs from the calculation
# print(wilcoxon(differences, zero_method='wilcox', alternative='less'))
#
# # print(f'Wilcoxon signed-rank test statistic: {stat}')
# # print(f'P-value: {p_value}')


import numpy as np
from scipy.stats import mannwhitneyu

# Data
benign_sizes = np.array([0.4, 2.1, 3.6, 0.6, 0.8, 2.4, 4.0])
malicious_sizes = np.array([1.2, 0.2, 0.3, 3.3, 2.0, 0.9, 1.1, 1.5])

# Perform the Mann-Whitney U test
stat, p_value = mannwhitneyu(benign_sizes, malicious_sizes, alternative='two-sided')

print(f'Mann-Whitney U statistic: {stat}')
print(f'P-value: {p_value}')
