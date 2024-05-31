from scipy.stats import sem, t
# from scipy import mean
from statistics import mean

confidence = 0.95
data = [10, 19, 11, 12, 15, 19, 9, 17, 1, 22, 9, 8]
n = len(data)
m = mean(data)
print("mean: ", m)

std_err = sem(data)

h = std_err * t.ppf((1 + confidence) / 2, n - 1)
start = m - h
end = m + h
print("start: ", start, " end: ", end)
