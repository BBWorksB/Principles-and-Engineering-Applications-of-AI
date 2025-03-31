import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import matplotlib.cm as cm

def generate_gmm_data(n_samples, means, covs, weights):
    """
    Generate synthetic data from a Gaussian Mixture Model.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    means : array-like, shape (n_components, n_features)
        Mean vectors for each Gaussian component
    covs : array-like, shape (n_components, n_features, n_features)
        Covariance matrices for each Gaussian component
    weights : array-like, shape (n_components,)
        Mixture weights for each component
        
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        Generated data points
    labels : array-like, shape (n_samples,)
        Component labels for each data point
    """
    # TODO: Implement GMM data generation
    # 1. Normalize weights to sum to 1
    # 2. Determine how many samples to generate from each component
    # 3. Generate samples from each Gaussian component
    # 4. Return the samples and their true component labels
    # Normalize weights to sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    # Determine the number of samples for each component
    n_components = len(weights)
    n_features = means.shape[1]
    samples_per_component = np.random.multinomial(n_samples, weights)
    
    # Generate samples for each component
    X = []
    labels = []
    for i in range(n_components):
        samples = np.random.multivariate_normal(means[i], covs[i], samples_per_component[i])
        X.append(samples)
        labels.extend([i] * samples_per_component[i])
    
    # Combine all samples and shuffle
    X = np.vstack(X)
    labels = np.array(labels)
    indices = np.random.permutation(n_samples)
    X = X[indices]
    labels = labels[indices]
    
    return X, labels
    
    # return None, None

def initialize_gmm_params(X, n_components, init_method='random'):
    """
    Initialize GMM parameters.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int
        Number of Gaussian components
    init_method : str
        Initialization method ('random' or 'kmeans')
        
    Returns:
    --------
    weights : array-like, shape (n_components,)
        Initial mixture weights
    means : array-like, shape (n_components, n_features)
        Initial component means
    covs : array-like, shape (n_components, n_features, n_features)
        Initial component covariance matrices
    """
    # TODO: Implement parameter initialization
    # 1. Initialize weights uniformly
    # 2. Initialize means either randomly or using k-means
    # 3. Initialize covariance matrices
    # Initialize weights uniformly
    weights = np.ones(n_components) / n_components

    # Initialize means
    if init_method == 'random':
        indices = np.random.choice(X.shape[0], n_components, replace=False)
        means = X[indices]
    elif init_method == 'kmeans':
        kmeans = KMeans(n_clusters=n_components, random_state=42)
        kmeans.fit(X)
        means = kmeans.cluster_centers_
    else:
        raise ValueError("Invalid initialization method. Choose 'random' or 'kmeans'.")

    # Initialize covariance matrices
    covs = np.array([np.cov(X, rowvar=False) for _ in range(n_components)])

    return weights, means, covs
    
    # return None, None, None

def compute_responsibilities(X, weights, means, covs):
    """
    E-step: Compute the posterior probabilities (responsibilities) of each component.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    weights : array-like, shape (n_components,)
        Mixture weights
    means : array-like, shape (n_components, n_features)
        Component means
    covs : array-like, shape (n_components, n_features, n_features)
        Component covariance matrices
        
    Returns:
    --------
    resp : array-like, shape (n_samples, n_components)
        Responsibilities of each component for each sample
    log_likelihood : float
        Log-likelihood of the data
    """
    # TODO: Implement the E-step of the EM algorithm
    # 1. Compute the weighted probability of each sample under each Gaussian component
    # 2. Normalize to get posterior probabilities (responsibilities)
    # 3. Compute the log-likelihood of the data
    n_samples, n_features = X.shape
    n_components = len(weights)
    
    # Compute the weighted probabilities for each component
    weighted_probs = np.zeros((n_samples, n_components))
    for k in range(n_components):
        rv = multivariate_normal(mean=means[k], cov=covs[k])
        weighted_probs[:, k] = weights[k] * rv.pdf(X)
    
    # Compute the log-likelihood
    log_likelihood = np.sum(np.log(np.sum(weighted_probs, axis=1)))
    
    # Normalize to get responsibilities
    resp = weighted_probs / np.sum(weighted_probs, axis=1, keepdims=True)
    
    return resp, log_likelihood
    
    # return None, None

def logsumexp(a, axis=None):
    """
    Compute log(sum(exp(a), axis=axis)) in a numerically stable way.
    """
    # TODO: Implement a numerically stable version of logsumexp
    # (Hint: Subtract the maximum value before taking exp to avoid overflow)
    a_max = np.max(a, axis=axis, keepdims=True)
    stable_exp = np.exp(a - a_max)
    sum_exp = np.sum(stable_exp, axis=axis, keepdims=True)
    log_sum_exp = np.log(sum_exp) + a_max
    if axis is not None:
        log_sum_exp = np.squeeze(log_sum_exp, axis=axis)
    return log_sum_exp
    
    # return None

def update_parameters(X, resp):
    """
    M-step: Update the GMM parameters using the current responsibilities.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    resp : array-like, shape (n_samples, n_components)
        Responsibilities
        
    Returns:
    --------
    weights : array-like, shape (n_components,)
        Updated mixture weights
    means : array-like, shape (n_components, n_features)
        Updated component means
    covs : array-like, shape (n_components, n_features, n_features)
        Updated component covariance matrices
    """
    # TODO: Implement the M-step of the EM algorithm
    # 1. Compute the effective number of points assigned to each component
    # 2. Update weights based on the sum of responsibilities
    # 3. Update means using weighted average of data points
    # 4. Update covariance matrices

    n_samples, n_features = X.shape
    n_components = resp.shape[1]
    
    # Compute the effective number of points assigned to each component
    N_k = np.sum(resp, axis=0)
    
    # Update weights
    weights = N_k / n_samples
    
    # Update means
    means = np.dot(resp.T, X) / N_k[:, np.newaxis]
    
    # Update covariance matrices
    covs = []
    for k in range(n_components):
        diff = X - means[k]
        cov = np.dot(resp[:, k] * diff.T, diff) / N_k[k]
        covs.append(cov)
    covs = np.array(covs)
    
    return weights, means, covs
    
    # return None, None, None

def fit_gmm(X, n_components, max_iter=100, tol=1e-3, init_method='random', verbose=False):
    """
    Fit a Gaussian Mixture Model to the data using the EM algorithm.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    n_components : int
        Number of Gaussian components
    max_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-3
        Convergence threshold for log-likelihood
    init_method : str, default='random'
        Initialization method ('random' or 'kmeans')
    verbose : bool, default=False
        Whether to print progress during fitting
        
    Returns:
    --------
    weights : array-like, shape (n_components,)
        Fitted mixture weights
    means : array-like, shape (n_components, n_features)
        Fitted component means
    covs : array-like, shape (n_components, n_features, n_features)
        Fitted component covariance matrices
    log_likelihood_history : list
        Log-likelihood at each iteration
    resp : array-like, shape (n_samples, n_components)
        Final responsibilities
    """
    # TODO: Implement the EM algorithm for fitting GMM
    # 1. Initialize parameters
    # 2. Iterate until convergence or max_iter:
    #    a. E-step: Compute responsibilities
    #    b. Check for convergence using log-likelihood
    #    c. M-step: Update parameters
    # Initialize parameters
    weights, means, covs = initialize_gmm_params(X, n_components, init_method)
    log_likelihood_history = []
    
    for iteration in range(max_iter):
        # E-step: Compute responsibilities
        resp, log_likelihood = compute_responsibilities(X, weights, means, covs)
        log_likelihood_history.append(log_likelihood)
        
        if verbose:
            print(f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood:.6f}")
        
        # Check for convergence
        if iteration > 0 and abs(log_likelihood - log_likelihood_history[-2]) < tol:
            if verbose:
                print("Convergence reached.")
            break
        
        # M-step: Update parameters
        weights, means, covs = update_parameters(X, resp)
    
    return weights, means, covs, log_likelihood_history, resp
    
    # return None, None, None, None, None

def plot_gmm_contours(X, weights, means, covs, ax=None, title=None):
    """
    Plot GMM contours over the data.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    weights : array-like, shape (n_components,)
        Mixture weights
    means : array-like, shape (n_components, n_features)
        Component means
    covs : array-like, shape (n_components, n_features, n_features)
        Component covariance matrices
    ax : matplotlib.axes.Axes, default=None
        Axes object to plot on
    title : str, default=None
        Plot title
    """
    # TODO: Implement visualization of GMM components
    # 1. Plot data points
    # 2. Plot component means
    # 3. Create contour plot to visualize the GMM density

    if ax is None:
        ax = plt.gca()
    
    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, label="Data points")
    
    # Plot component means
    ax.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label="Component means")
    
    # Create contour plot for each Gaussian component
    x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    for k in range(len(weights)):
        rv = multivariate_normal(mean=means[k], cov=covs[k])
        Z = rv.pdf(grid_points).reshape(X_grid.shape)
        ax.contour(X_grid, Y_grid, Z, levels=5, cmap=cm.get_cmap('viridis'), alpha=0.7)
    
    # Add title and legend
    if title:
        ax.set_title(title)
    ax.legend()
    
    # return None

def main():
    """
    Main function to demonstrate GMM parameter learning with EM algorithm.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # TODO: Define true parameters and generate synthetic data
    true_means = np.array([[0, 0], [3, 3], [0, 4]])
    true_covs = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]]])
    true_weights = np.array([0.4, 0.4, 0.2])
    n_samples = 500

    # Generate synthetic data
    X, true_labels = generate_gmm_data(n_samples, true_means, true_covs, true_weights)

    # TODO: Fit GMM with EM algorithm
    n_components = 3
    weights, means, covs, log_likelihood_history, resp = fit_gmm(
        X, n_components, max_iter=100, tol=1e-3, init_method='kmeans', verbose=True
    )

    # TODO: Visualize results and analyze
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_gmm_contours(X, weights, means, covs, ax=ax, title="Fitted GMM")
    plt.show()
    
    pass

if __name__ == "__main__":
    main()