# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# import matplotlib.cm as cm

# def generate_gmm_data(n_samples, means, covs, weights):
#     """
#     Generate synthetic data from a Gaussian Mixture Model.
    
#     Parameters:
#     -----------
#     n_samples : int
#         Number of samples to generate
#     means : array-like, shape (n_components, n_features)
#         Mean vectors for each Gaussian component
#     covs : array-like, shape (n_components, n_features, n_features)
#         Covariance matrices for each Gaussian component
#     weights : array-like, shape (n_components,)
#         Mixture weights for each component
        
#     Returns:
#     --------
#     X : array-like, shape (n_samples, n_features)
#         Generated data points
#     labels : array-like, shape (n_samples,)
#         Component labels for each data point
#     """
#     # TODO: Implement GMM data generation
#     # 1. Normalize weights to sum to 1
#     # 2. Determine how many samples to generate from each component
#     # 3. Generate samples from each Gaussian component
#     # 4. Return the samples and their true component labels
    
#     return None, None

# def initialize_gmm_params(X, n_components, init_method='random'):
#     """
#     Initialize GMM parameters.
    
#     Parameters:
#     -----------
#     X : array-like, shape (n_samples, n_features)
#         Training data
#     n_components : int
#         Number of Gaussian components
#     init_method : str
#         Initialization method ('random' or 'kmeans')
        
#     Returns:
#     --------
#     weights : array-like, shape (n_components,)
#         Initial mixture weights
#     means : array-like, shape (n_components, n_features)
#         Initial component means
#     covs : array-like, shape (n_components, n_features, n_features)
#         Initial component covariance matrices
#     """
#     # TODO: Implement parameter initialization
#     # 1. Initialize weights uniformly
#     # 2. Initialize means either randomly or using k-means
#     # 3. Initialize covariance matrices
    
#     return None, None, None

# def compute_responsibilities(X, weights, means, covs):
#     """
#     E-step: Compute the posterior probabilities (responsibilities) of each component.
    
#     Parameters:
#     -----------
#     X : array-like, shape (n_samples, n_features)
#         Training data
#     weights : array-like, shape (n_components,)
#         Mixture weights
#     means : array-like, shape (n_components, n_features)
#         Component means
#     covs : array-like, shape (n_components, n_features, n_features)
#         Component covariance matrices
        
#     Returns:
#     --------
#     resp : array-like, shape (n_samples, n_components)
#         Responsibilities of each component for each sample
#     log_likelihood : float
#         Log-likelihood of the data
#     """
#     # TODO: Implement the E-step of the EM algorithm
#     # 1. Compute the weighted probability of each sample under each Gaussian component
#     # 2. Normalize to get posterior probabilities (responsibilities)
#     # 3. Compute the log-likelihood of the data
    
#     return None, None

# def logsumexp(a, axis=None):
#     """
#     Compute log(sum(exp(a), axis=axis)) in a numerically stable way.
#     """
#     # TODO: Implement a numerically stable version of logsumexp
#     # (Hint: Subtract the maximum value before taking exp to avoid overflow)
    
#     return None

# def update_parameters(X, resp):
#     """
#     M-step: Update the GMM parameters using the current responsibilities.
    
#     Parameters:
#     -----------
#     X : array-like, shape (n_samples, n_features)
#         Training data
#     resp : array-like, shape (n_samples, n_components)
#         Responsibilities
        
#     Returns:
#     --------
#     weights : array-like, shape (n_components,)
#         Updated mixture weights
#     means : array-like, shape (n_components, n_features)
#         Updated component means
#     covs : array-like, shape (n_components, n_features, n_features)
#         Updated component covariance matrices
#     """
#     # TODO: Implement the M-step of the EM algorithm
#     # 1. Compute the effective number of points assigned to each component
#     # 2. Update weights based on the sum of responsibilities
#     # 3. Update means using weighted average of data points
#     # 4. Update covariance matrices
    
#     return None, None, None

# def fit_gmm(X, n_components, max_iter=100, tol=1e-3, init_method='random', verbose=False):
#     """
#     Fit a Gaussian Mixture Model to the data using the EM algorithm.
    
#     Parameters:
#     -----------
#     X : array-like, shape (n_samples, n_features)
#         Training data
#     n_components : int
#         Number of Gaussian components
#     max_iter : int, default=100
#         Maximum number of EM iterations
#     tol : float, default=1e-3
#         Convergence threshold for log-likelihood
#     init_method : str, default='random'
#         Initialization method ('random' or 'kmeans')
#     verbose : bool, default=False
#         Whether to print progress during fitting
        
#     Returns:
#     --------
#     weights : array-like, shape (n_components,)
#         Fitted mixture weights
#     means : array-like, shape (n_components, n_features)
#         Fitted component means
#     covs : array-like, shape (n_components, n_features, n_features)
#         Fitted component covariance matrices
#     log_likelihood_history : list
#         Log-likelihood at each iteration
#     resp : array-like, shape (n_samples, n_components)
#         Final responsibilities
#     """
#     # TODO: Implement the EM algorithm for fitting GMM
#     # 1. Initialize parameters
#     # 2. Iterate until convergence or max_iter:
#     #    a. E-step: Compute responsibilities
#     #    b. Check for convergence using log-likelihood
#     #    c. M-step: Update parameters
    
#     return None, None, None, None, None

# def plot_gmm_contours(X, weights, means, covs, ax=None, title=None):
#     """
#     Plot GMM contours over the data.
    
#     Parameters:
#     -----------
#     X : array-like, shape (n_samples, n_features)
#         Data points
#     weights : array-like, shape (n_components,)
#         Mixture weights
#     means : array-like, shape (n_components, n_features)
#         Component means
#     covs : array-like, shape (n_components, n_features, n_features)
#         Component covariance matrices
#     ax : matplotlib.axes.Axes, default=None
#         Axes object to plot on
#     title : str, default=None
#         Plot title
#     """
#     # TODO: Implement visualization of GMM components
#     # 1. Plot data points
#     # 2. Plot component means
#     # 3. Create contour plot to visualize the GMM density
    
#     return None

# def main():
#     """
#     Main function to demonstrate GMM parameter learning with EM algorithm.
#     """
#     # Set random seed for reproducibility
#     np.random.seed(42)
    
#     # TODO: Define true parameters and generate synthetic data
    
#     # TODO: Fit GMM with EM algorithm
    
#     # TODO: Visualize results and analyze
    
#     pass

# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
from scipy.special import logsumexp
from sklearn.cluster import KMeans

def generate_gmm_data(n_samples, means, covs, weights):
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    n_components = len(means)
    
    # Sample component assignments
    component_samples = np.random.choice(n_components, size=n_samples, p=weights)
    
    X = np.vstack([np.random.multivariate_normal(means[i], covs[i], (component_samples == i).sum())
                   for i in range(n_components)])
    labels = np.concatenate([[i] * (component_samples == i).sum() for i in range(n_components)])
    
    return X, labels

def initialize_gmm_params(X, n_components, init_method='random'):
    n_samples, n_features = X.shape
    weights = np.ones(n_components) / n_components
    
    if init_method == 'kmeans':
        kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=42).fit(X)
        means = kmeans.cluster_centers_
    else:  # Random initialization
        means = X[np.random.choice(n_samples, n_components, replace=False)]
    
    covs = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(n_components)])
    
    return weights, means, covs

def compute_responsibilities(X, weights, means, covs):
    n_samples, n_components = X.shape[0], len(weights)
    log_probs = np.zeros((n_samples, n_components))
    
    for i in range(n_components):
        log_probs[:, i] = np.log(weights[i]) + multivariate_normal.logpdf(X, means[i], covs[i])
    
    log_likelihood = np.sum(logsumexp(log_probs, axis=1))
    responsibilities = np.exp(log_probs - logsumexp(log_probs, axis=1, keepdims=True))
    
    return responsibilities, log_likelihood

def update_parameters(X, resp):
    n_samples, n_features = X.shape
    n_components = resp.shape[1]
    
    Nk = resp.sum(axis=0)  # Effective number of points per component
    weights = Nk / n_samples
    means = (resp.T @ X) / Nk[:, np.newaxis]
    
    covs = np.zeros((n_components, n_features, n_features))
    for i in range(n_components):
        diff = X - means[i]
        covs[i] = (resp[:, i][:, np.newaxis] * diff).T @ diff / Nk[i] + 1e-6 * np.eye(n_features)
    
    return weights, means, covs

def fit_gmm(X, n_components, max_iter=100, tol=1e-3, init_method='random', verbose=False):
    weights, means, covs = initialize_gmm_params(X, n_components, init_method)
    log_likelihood_history = []
    
    for iteration in range(max_iter):
        resp, log_likelihood = compute_responsibilities(X, weights, means, covs)
        log_likelihood_history.append(log_likelihood)
        
        if iteration > 0 and abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < tol:
            break
        
        weights, means, covs = update_parameters(X, resp)
        if verbose:
            print(f"Iteration {iteration+1}: Log-Likelihood = {log_likelihood:.4f}")
    
    return weights, means, covs, log_likelihood_history, resp

def plot_gmm_contours(X, weights, means, covs, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
    
    colors = cm.viridis(np.linspace(0, 1, len(means)))
    x, y = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                        np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    xy = np.column_stack([x.ravel(), y.ravel()])
    
    for mean, cov, color in zip(means, covs, colors):
        z = multivariate_normal(mean, cov).pdf(xy).reshape(100, 100)
        ax.contour(x, y, z, levels=5, colors=[color])
        ax.scatter(*mean, color=color, s=100, edgecolors='k')
    
    if title:
        ax.set_title(title)
    
    plt.show()

def main():
    np.random.seed(42)
    true_means = np.array([[2, 2], [-2, -2], [2, -2]])
    true_covs = np.array([np.eye(2) * 0.5, np.eye(2) * 0.8, np.eye(2) * 0.6])
    true_weights = np.array([0.4, 0.4, 0.2])
    
    X, labels = generate_gmm_data(500, true_means, true_covs, true_weights)
    weights, means, covs, log_likelihood_history, resp = fit_gmm(X, 3, verbose=True)
    
    plot_gmm_contours(X, weights, means, covs, title="Fitted GMM")

if __name__ == "__main__":
    main()
