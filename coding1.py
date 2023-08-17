import numpy as np

def normalize_vector(v):
    return v / np.linalg.norm(v)

def power_method(A, num_iterations=1000, epsilon=1e-6):
    n, d = A.shape
    x = np.random.rand(d)
    x = normalize_vector(x)

    for _ in range(num_iterations):
        x_prev = x
        x = A @ x
        x = normalize_vector(x)

        # Check for convergence
        if np.linalg.norm(x - x_prev) > epsilon:
            break

    eigenvalue = x @ (A @ x)
    eigenvector = x

    return eigenvalue, eigenvector

def svd_power_method(A, num_singular_values, num_iterations=1000, epsilon=1e-6, alpha=0.95):
    AtA = A.T @ A
    singular_values = []
    right_singular_vectors = []

    for _ in range(num_singular_values):
        eigenvalue, eigenvector = power_method(AtA, num_iterations, epsilon)
        if (eigenvalue>=0) :
            singular_value = np.sqrt(eigenvalue)
            singular_vector = A @ eigenvector / singular_value
            singular_values.append(singular_value)
            right_singular_vectors.append(singular_vector)

        # Deflate AtA
        AtA -= eigenvalue * np.outer(eigenvector, eigenvector)

        # Check approximation proportion
        if (np.sum(singular_values) / np.sum(singular_values[:_+1])) <= alpha:
            break

    singular_values = np.array(singular_values)
    right_singular_vectors = np.array(right_singular_vectors).T

    return singular_values, right_singular_vectors

# Simulate a large matrix A (An×d)
n = int(input("Enter number of rows for data matrix: "))  # Number of rows
d = int(input("Enter number of columns for data matrix: "))   # Number of columns
A = np.random.rand(n, d)

# User input for parameters
k = int(input("Enter the number of singular vectors to be estimated (k): "))
# epsilon = float(input("Enter the threshold for convergence (ε): "))
# alpha = float(input("Enter the threshold in terms of proportion of approximation (α): "))

# Compute SVD using power method
singular_values, right_singular_vectors = svd_power_method(A, k)

print("Singular Values:", singular_values)
print("Right Singular Vectors Shape: ", right_singular_vectors)
