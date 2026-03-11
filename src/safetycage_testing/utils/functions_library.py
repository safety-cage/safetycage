# This script contains the library of functions used in the Exaigon safety_cage development
import numpy as np
from scipy.stats import cauchy, multivariate_normal

def CauchyCombinationTest(p_values, weights=None):

    # If weights is None, put equal weight to each p-value:
    if weights is None or weights == []:
        weights = np.ones(len(p_values))/len(p_values)

    # Compute Cauchy statistic:
    C = np.sum(weights*np.tan((0.5-p_values)*np.pi))

    # If p-value are uniformly distributed, C has a standard Cauchy distribution
    # Small p-values indicate discrepancies from H_0, which will give large C.
    # Compute one-sided right-tailed p-value:
    p_value_combined_cauchy = 1 - cauchy.cdf(C, loc=0, scale=1)

    return(p_value_combined_cauchy)



def fastSPARDA(X_samples, Y_samples, **kwargs):
    #This method directly solves the original nonconvex formulation using subgradient
    # hill-climbing (with l1 penalty on the projection vector).
    # Thus it is more efficient, but results may heavily depend on 
    # initialization if the underlying distributions induce nonconvexity in
    # the SPARDA objective function.

    # Parse input arguments
    lambdas_default = [0]
    num_folds_default = 5
    max_iter_default = 1000
    eps_default = 1e-8
    learning_rate_default = 1
    print_update_default = 100

    lambdas = kwargs.get('lambdas', lambdas_default)
    num_folds = kwargs.get('num_folds', num_folds_default)
    max_iter = kwargs.get('max_iter', max_iter_default)
    eps = kwargs.get('eps', eps_default)
    learning_rate = kwargs.get('learning_rate', learning_rate_default)
    print_update = kwargs.get('print_update', print_update_default)

    if X_samples.shape[0] < Y_samples.shape[0]:#ensure n >= m with dim(X)=n,dim(y)=m by swapping
        X_samples, Y_samples = Y_samples, X_samples

    n, d = X_samples.reshape(-1), X_samples.shape[1]
    m = Y_samples.shape[0]

    x_foldsize = n // num_folds
    y_foldsize = m // num_folds
    prev_beta = np.zeros(d)
    prev_beta[0] = 0.5

    if len(lambdas) > 1:
        lambdas = sorted(lambdas)
        lambda_scores = np.zeros(len(lambdas))
        first_beta, first_cost = l1SPARDA(X_samples, Y_samples, 0, max_iter, eps, learning_rate, print_update, prev_beta)

        for fold in range(1, num_folds + 1):
        
            x_foldindex = (fold-1) * x_foldsize + 1
            y_foldindex = (fold-1) * y_foldsize + 1
            if fold < num_folds:
                xs_test = X_samples[x_foldindex:(x_foldindex+x_foldsize-1),:]
                xs_train = X_samples[1:(x_foldindex-1) (x_foldindex+x_foldsize):n]
                ys_test = Y_samples[y_foldindex:(y_foldindex+y_foldsize-1),:]
                ys_train = Y_samples[1:(y_foldindex-1) (y_foldindex+y_foldsize):m]              
            else:
                xs_test = X_samples[x_foldindex:n,:]
                xs_train = X_samples[1:(x_foldindex-1),:]
                ys_test = Y_samples[y_foldindex:m,:]
                ys_train = Y_samples[1:(y_foldindex-1),:]
            


            prev_beta = first_beta
            for l in range(len(lambdas)):
                lambda_val = lambdas[l]
                beta, cost = l1SPARDA(xs_train, ys_train, lambda_val, max_iter, eps, learning_rate, print_update, prev_beta)

                if np.linalg.norm(beta) > 0:
                    prev_beta = beta / np.linalg.norm(beta)

                heldout_wass = projectedWasserstein(xs_test, ys_test, beta)
                lambda_scores[l] += heldout_wass

                cardinality = np.sum(np.abs(beta) > 0)

                if print_update < np.inf:
                    print(f'lambda: {lambda_val}  fold: {fold}')
                    print(f'training cost: {cost}')
                    print(f'heldout cost: {heldout_wass}')
                    print(f'projection cardinality: {cardinality}')

                if cardinality < 2:
                    break

        lambda_scores /= num_folds
        best_indices = np.where(lambda_scores == np.max(lambda_scores))[0]
        best_lambda = lambdas[best_indices[0]]
    elif len(lambdas) == 1:
        best_lambda = lambdas[0]
        prev_beta, _, _ = randomProjectionSearch(X_samples, Y_samples, max_iter=max(100, int(np.ceil(max_iter / 10))))
    else:
        raise ValueError('lambdas not correctly formatted')

    beta_hat, cost = l1SPARDA(X_samples, Y_samples, best_lambda, max_iter, eps, learning_rate, print_update, prev_beta)
    beta_hat = beta_hat/np.linalg.norm(beta_hat) # always re-scale to unit norm.
    wass_dist = projectedWasserstein(X_samples, Y_samples, beta_hat)

    return beta_hat, wass_dist, cost, best_lambda

def l1SPARDA(xs, ys, _lambda, max_iter, eps, learning_rate, print_update, beta0=None):
    

    n, d = xs.shape
    m = ys.shape[0]
    iter_val = 0

    if beta0 is None:
        beta0 = np.sqrt(d) / d / 2 * np.ones(d)

    beta = beta0
    last_cost = -np.inf
    cost = projectedWasserstein(xs, ys, beta) - _lambda * np.sum(np.abs(beta))
    grad = np.zeros(d)

    while iter_val < max_iter and cost - last_cost >= eps:
        last_cost = cost
        last_beta = beta
        iter_val += 1
        step_size = learning_rate / np.sqrt(iter_val)

        if print_update > 0 and iter_val % print_update == 0:
            print(f'iter: {iter_val}   cost: {cost}  grad-norm: {np.linalg.norm(grad)}    beta_norm: {np.linalg.norm(beta)}  Step-size: {step_size}')

        # Compute gradient:
        projected_xs, x_order = np.sort(xs @ beta), np.argsort(xs @ beta)
        projected_ys, y_order = np.sort(ys @ beta), np.argsort(ys @ beta)
        grad = np.zeros(d)
        quant_x = 0
        quant_y = 0
        last_quant = 0
        NUMERIC_FACTOR = 1e-6
        delta = NUMERIC_FACTOR / (n * m)

        x_index = 0
        y_index = 0

        while quant_x < 1 - delta or quant_y < 1 - delta:
            next_quant_x = quant_x + 1 / n
            next_quant_y = quant_y + 1 / m
            proj_x = projected_xs[x_index]
            proj_y = projected_ys[y_index]

            while next_quant_x < next_quant_y - delta:
                grad += 2 * (proj_x - proj_y) * (xs[x_order[x_index], :] - ys[y_order[y_index], :]).T * (next_quant_x - last_quant)
                quant_x = next_quant_x
                last_quant = quant_x
                next_quant_x = quant_x + 1 / n
                x_index = min(x_index + 1, n - 1)
                proj_x = projected_xs[x_index]

            if quant_x < 1 - delta or quant_y < 1 - delta:
                if abs(next_quant_x - next_quant_y) < delta:
                    grad += 2 * (proj_x - proj_y) * (xs[x_order[x_index], :] - ys[y_order[y_index], :]).T * (next_quant_x - last_quant)
                    quant_x = next_quant_x
                    quant_y = next_quant_y
                    x_index = min(x_index + 1, n - 1)
                    y_index = min(y_index + 1, m - 1)
                    #x_index = x_index + 1
                    #y_index = y_index + 1              
                    last_quant = quant_x
                else:
                    grad += 2 * (proj_x - proj_y) * (xs[x_order[x_index], :] - ys[y_order[y_index], :]).T * (next_quant_y - last_quant)
                    quant_y = next_quant_y
                    y_index = min(y_index + 1, m - 1)
                    last_quant = quant_y

        MAX_BACKTRACKING = 10
        backtrack_tries = 0

        while cost - last_cost < eps and backtrack_tries <= MAX_BACKTRACKING:
            step_size = learning_rate / np.log(iter_val + 1) * 2 ** (-backtrack_tries)
            beta += step_size * grad

            if beta[0] < 0:
                beta = -beta

            if _lambda > 0:
                beta = np.sign(beta) * np.maximum(np.abs(beta) - _lambda * step_size, 0)

            beta_norm = np.linalg.norm(beta)

            if beta_norm > 1:
                beta /= beta_norm

            cost = projectedWasserstein(xs, ys, beta) - _lambda * np.sum(np.abs(beta))
            backtrack_tries += 1

        if cost < last_cost:
            cost = last_cost
            beta = last_beta
            break

    if iter_val >= max_iter:
        print('fastSPARDA optimization failed to converge (likely max_iter or learning_rate is too small)')

    return beta, cost



def randomProjectionSearch(X_samples, Y_samples, **kwargs):
    # Simple random search to find projection BETA which maximizes projected Wasserstein distance.
    # Optional arguments: max_iter = the number of random projections to try.

    if X_samples.shape[1] != Y_samples.shape[1]:
        raise ValueError('X and Y must have the same dimension')

    # Parse input arguments
    max_iter_default = 100
    max_iter = kwargs.get('max_iter', max_iter_default)

    best_dist = -np.inf
    best_beta = np.nan

    for i in range(1, max_iter + 1):
        beta = multivariate_normal.rvs(mean=np.zeros(X_samples.shape[1]), cov=np.eye(X_samples.shape[1]))
        beta /= np.linalg.norm(beta)

        if beta[0] < 0:
            beta = -beta

        d_beta = projectedWasserstein(X_samples, Y_samples, beta)

        if d_beta > best_dist:
            best_dist = d_beta
            best_beta = beta

    # Also check basis directions:
    for l in range(X_samples.shape[1]):
        beta = np.zeros(X_samples.shape[1])
        beta[l] = 1
        d_beta = projectedWasserstein(X_samples, Y_samples, beta)

        if d_beta > best_dist:
            best_dist = d_beta
            best_beta = beta

    param_settings = {'max_iter': max_iter}
    
    return best_beta, best_dist, param_settings


def projectedWasserstein(X_samples, Y_samples, beta):
    # Projected SQUARED empirical Wasserstein distance in direction beta
    # between X_samples and Y_samples.

    NUMERIC_FACTOR = 1e6  # used in ceil function to get the index of the quantile in the projected data vector.

    projected_xs = np.sort(X_samples.dot(beta))
    projected_ys = np.sort(Y_samples.dot(beta))
    
    if len(projected_xs) < len(projected_ys):
        projected_xs, projected_ys = projected_ys, projected_xs

    n = len(projected_xs)
    m = len(projected_ys)
    eps = 1 / (n * m * NUMERIC_FACTOR)
    dist = 0
    quant_x = 0
    quant_y = 0
    last_quant = 0
    x_index = 0
    y_index = 0
    while quant_x < 1 - eps or quant_y < 1 - eps:
        next_quant_x = quant_x + 1 / n
        next_quant_y = quant_y + 1 / m
        proj_x = projected_xs[x_index] if x_index < n else float('inf')
        proj_y = projected_ys[y_index] if y_index < m else float('inf')

        while next_quant_x < next_quant_y - eps:
            dist += (proj_x - proj_y) ** 2 * (next_quant_x - last_quant)
            quant_x = next_quant_x
            last_quant = quant_x
            next_quant_x = quant_x + 1 / n
            x_index = min(x_index + 1, n)
            proj_x = projected_xs[x_index] if x_index < n else float('inf')

        if quant_x < 1 - eps or quant_y < 1 - eps:
            if abs(next_quant_x - next_quant_y) < eps:
                dist += (proj_x - proj_y) ** 2 * (next_quant_x - last_quant)
                quant_x = next_quant_x
                quant_y = next_quant_y
                x_index += 1
                y_index += 1
                last_quant = quant_x
            else:
                dist += (proj_x - proj_y) ** 2 * (next_quant_y - last_quant)
                quant_y = next_quant_y
                y_index = min(y_index + 1, m)
                last_quant = quant_y

    return dist




def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


