import numpy as np
import sys
sys.path.append('.')
from FisherForecast import load_mock_samples

samples = load_mock_samples(90, n_realizations=1000)
pdf_data = samples['pdf']

# Check sum
sums = np.sum(pdf_data, axis=1)
print("Mean sum:", np.mean(sums))
print("Std sum:", np.std(sums))

cov = np.cov(pdf_data, rowvar=False)
try:
    inv = np.linalg.inv(cov)
    print("Inversion successful")
except np.linalg.LinAlgError:
    print("Matrix is singular")

eigvals = np.linalg.eigvalsh(cov)
print("Min eigenvalue:", np.min(eigvals))
print("Max eigenvalue:", np.max(eigvals))
print("Condition number:", np.max(eigvals)/np.min(eigvals))

# Check correlation summing
sigma = np.sqrt(np.diag(cov))
corr = cov / np.outer(sigma, sigma)
print("Max off-diagonal correlation:", np.max(corr - np.eye(len(corr))))
