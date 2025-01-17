#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from copy import deepcopy
import numpy as np
from astropy.stats import sigma_clip
from scipy import special
from scipy.sparse import spdiags
from scipy import stats
from scipy import odr

#region Classes

class FitDetails:
    pass

class FitOptions:
    def __init__(self):
        self.exclude_high_leverage = False
        self.exclude_outliers = False
        self.force_use_scaling = False
        self.include_statistics = False
        self.method = ''

class FitException(Exception):
    pass

#endregion

#region Fitting: Solvers

# pylint: disable=invalid-unary-operand-type
def solve_linear_regression(X, y, sigma = None, excluded = None, options = None):
    ##############################
    # Parameters                 #
    ##############################

    N = len(y)

    if needs_reshaping(N, X) or needs_reshaping(N, y):
        X, y, _, sigma, excluded = reshape_data(N, X, y, None, sigma, excluded)

    if sigma is None or (not hasattr(sigma, '__len__') and sigma == 0.0) or (hasattr(sigma, '__len__') and len(sigma) > 1 and len(sigma) != N):
        sigma = 1

    if not hasattr(sigma, '__len__') or len(sigma) == 1:
        sigma = sigma * np.ones(y.shape)

    if excluded is None or not hasattr(excluded, '__len__') or len(excluded) != N:
        excluded = np.zeros(N, dtype=bool)

    if options is None:
        exclude_high_leverage = False
        exclude_outliers = False
        exclude_outliers_num_sigma = 3
        include_statistics = False
        relative = False
    else:
        exclude_high_leverage = hasattr(options, 'exclude_high_leverage') and options.exclude_high_leverage
        exclude_outliers = hasattr(options, 'exclude_outliers') and options.exclude_outliers
        exclude_outliers_num_sigma = options.exclude_outliers_num_sigma if hasattr(options, 'exclude_outliers_num_sigma') else 3
        include_statistics = hasattr(options, 'include_statistics') and options.include_statistics
        relative = hasattr(options, 'relative') and options.relative

    weighted = np.any(sigma != 1.0)
    if weighted:
        method = 'WOLS'
    else:
        method = 'OLS'

    ##############################
    # Solve Linear Least Squares #
    ##############################

    n = 1
    if exclude_outliers:
        n += 1
    if exclude_high_leverage:
        n += 1
        if exclude_outliers:
            n += 1

    i = 0
    while i < n:
        XFit = X[~excluded,:]
        yFit = y[~excluded]
        sigmaFit = sigma[~excluded]

        Nfit = len(yFit)
        df = Nfit - len(XFit[0])
        if df <= 0:
            return None

        if relative:
            sigmaFit = yFit
            D = spdiags(np.transpose(np.power(sigmaFit,-1)), 0, Nfit, Nfit)
            b = D*yFit
            A = D*XFit
        else:
            b = yFit/sigmaFit
            A = XFit/sigmaFit                                       # Design Matrix

        U, WD, VH = np.linalg.svd(A, full_matrices=False)           # Singular Value Decomposition
        W = np.diag(WD)
        V = np.transpose(VH)

        eps = np.finfo(y.dtype).eps                                 # Handle singular values
        Winv = W.copy()
        Winv[Winv != 0] = 1/W[W != 0]
        Winv[W < Nfit*eps] = 0

        a = np.matmul(np.matmul(np.matmul(V, Winv), np.transpose(U)), b)
        C = np.reshape(np.matmul(np.matmul(V, np.power(Winv,2)), np.transpose(V)), (len(a),len(a)))

        if exclude_outliers and (i == 0 or i == 2):
            r = yFit - np.reshape(np.matmul(XFit, a), yFit.shape)
            excluded[~excluded] = is_outlier(r, num_sigma = exclude_outliers_num_sigma).flatten()
        elif exclude_high_leverage and (i == 0 or i == 1):
            # See: https://www.mathworks.com/help/stats/leverage.html
            # See: https://en.wikipedia.org/wiki/Leverage_(statistics)
            # See: https://online.stat.psu.edu/stat501/book/export/html/973
            Q, _ = np.linalg.qr(A)
            leverageThreshold = 3 * len(a) / Nfit
            leverage = np.transpose(np.sum(np.transpose(Q)*np.transpose(Q), axis=0))   # Same as: h=diag(H) where H=Q*Q'
            excluded[~excluded] = leverage > leverageThreshold

        i += 1

    details = FitDetails()
    details.fit_type = method
    details.fit_weighted = weighted
    details.success = not np.any(np.isnan(a))

    details.M = len(a)                                              # Number of Coefficients
    details.N = Nfit                                                # Number of Included Points
    details.NExcl = N - Nfit                                        # Number of Excluded Points
    details.df = df                                                 # Degrees of Freedom

    details.coeff = a                                               # Coefficients
    details.cov = C                                                 # Covariance Matrix

    append_fit_statistics(details, X, y, sigma, excluded, include_statistics)

    details.coeffVar = np.reshape(np.diag(C), details.coeff.shape)  # Coefficient Variance
    details.coeffUnc = np.sqrt(details.rchi2*details.coeffVar)      # Coefficient Uncertainty (68% confidence interval)

    return details

def solve_linear_regression_error_xy(x, y, sigma_x, sigma_y, excluded = None, options = None, coeff0 = None, fix_coeff = None):
    ##############################
    # Parameters                 #
    ##############################

    N = len(y)

    if needs_reshaping(N, x) or needs_reshaping(N, y):
        x, y, sigma_x, sigma_y, excluded = reshape_data(N, x, y, sigma_x, sigma_y, excluded)

    if sigma_x is not None:
        if (not hasattr(sigma_x, '__len__') and sigma_x == 0.0) or (hasattr(sigma_x, '__len__') and len(sigma_x) > 1 and len(sigma_x) != N):
            sigma_x = None
        elif not hasattr(sigma_x, '__len__') or len(sigma_x) == 1:
            sigma_x = sigma_x * np.ones(y.shape)

    if sigma_y is not None:
        if (not hasattr(sigma_y, '__len__') and sigma_y == 0.0) or (hasattr(sigma_y, '__len__') and len(sigma_y) > 1 and len(sigma_y) != N):
            sigma_y = None
        elif not hasattr(sigma_y, '__len__') or len(sigma_y) == 1:
            sigma_y = sigma_y * np.ones(y.shape)

    if excluded is None or not hasattr(excluded, '__len__') or len(excluded) != N:
        excluded = np.zeros(N, dtype=bool)

    weighted = True
    method = 'ODR'

    if options is None:
        exclude_high_leverage = False
        exclude_outliers = False
        exclude_outliers_num_sigma = 3
        force_use_scaling = False
        include_statistics = False
    else:
        exclude_high_leverage = hasattr(options, 'exclude_high_leverage') and options.exclude_high_leverage
        exclude_outliers = hasattr(options, 'exclude_outliers') and options.exclude_outliers
        exclude_outliers_num_sigma = options.exclude_outliers_num_sigma if hasattr(options, 'exclude_outliers_num_sigma') else 3
        force_use_scaling = hasattr(options, 'force_use_scaling') and options.force_use_scaling
        include_statistics = hasattr(options, 'include_statistics') and options.include_statistics
        if hasattr(options, 'method') and options.method != '':
            method = options.method

    ##############################
    # Prepare Data               #
    ##############################

    use_scaling = force_use_scaling
    if use_scaling:
        x, xMean, xStd = scale_x(x, excluded)

    X = np.hstack((np.resize(1.,x.shape), x))

    ##############################
    # Solve Problem              #
    ##############################

    n = 1
    if exclude_outliers:
        n += 1
    if exclude_high_leverage:
        n += 1
        if exclude_outliers:
            n += 1

    i = 0
    while i < n:
        XFit = X[~excluded,:]
        xFit = x[~excluded]
        yFit = y[~excluded]
        sigma_xFit = sigma_x[~excluded] if sigma_x is not None else None
        sigma_yFit = sigma_y[~excluded] if sigma_y is not None else None

        Nfit = len(yFit)
        df = Nfit - len(XFit[0])
        if df <= 0:
            return None

        if coeff0 is not None and not np.any(np.logical_not(fix_coeff)):
            a = np.reshape(coeff0, (len(coeff0),1))
            a_unc = np.zeros((len(coeff0),1))
            C = np.zeros((len(coeff0),len(coeff0)))
            rchi2 = 0.0
        else:
            match method:
                case 'ODR':
                    odr_data = odr.RealData(xFit.flatten(), yFit.flatten(), sx=sigma_xFit.flatten() if sigma_xFit is not None else None, sy=sigma_yFit.flatten() if sigma_yFit is not None else None)
                    odr_fit = odr.ODR(odr_data, odr.unilinear, beta0=np.flip(coeff0) if coeff0 is not None else None, ifixb=np.flip(np.logical_not(fix_coeff)) if coeff0 is not None and fix_coeff is not None else None)
                    odr_output = odr_fit.run()

                    a = np.reshape(np.flip(odr_output.beta), (len(odr_output.beta),1))
                    a_unc = np.reshape(np.flip(odr_output.sd_beta), (len(odr_output.beta),1))
                    C = np.flip(odr_output.cov_beta)
                    non_zero_index = np.argmax(a_unc != 0.0)
                    rchi2 = a_unc[non_zero_index]**2 / C[non_zero_index,non_zero_index]

        if exclude_outliers and (i == 0 or i == 2):
            r = yFit - np.reshape(np.matmul(XFit, a), yFit.shape)
            excluded[~excluded] = is_outlier(r, num_sigma = exclude_outliers_num_sigma).flatten()
        elif exclude_high_leverage and (i == 0 or i == 1):
            r = yFit - np.reshape(np.matmul(XFit, a), yFit.shape)
            sigmaFit = np.std(r)
            A = XFit/sigmaFit

            # See: https://www.mathworks.com/help/stats/leverage.html
            # See: https://en.wikipedia.org/wiki/Leverage_(statistics)
            # See: https://online.stat.psu.edu/stat501/book/export/html/973
            Q, _ = np.linalg.qr(A)
            leverageThreshold = 3 * len(a) / Nfit
            leverage = np.transpose(np.sum(np.transpose(Q)*np.transpose(Q), axis=0))   # Same as: h=diag(H) where H=Q*Q'
            excluded[~excluded] = leverage > leverageThreshold

        i += 1

    details = FitDetails()
    details.fit_type = method
    details.fit_weighted = weighted
    details.success = not np.any(np.isnan(a))

    details.M = len(a)                                              # Number of Coefficients
    details.N = Nfit                                                # Number of Included Points
    details.NExcl = N - Nfit                                        # Number of Excluded Points
    details.df = df                                                 # Degrees of Freedom

    details.coeff = a                                               # Coefficients
    details.cov = C                                                 # Covariance Matrix
    details.rchi2 = rchi2                                           # Update RChi2 to match that used by ODR

    append_fit_statistics(details, X, y, None, excluded, include_statistics)

    if use_scaling:
        unscale_coefficients(details, xMean, xStd)

    details.coeffVar = np.reshape(np.diag(C), details.coeff.shape)  # Coefficient Variance
    details.coeffUnc = np.sqrt(details.rchi2*details.coeffVar)      # Coefficient Uncertainty (68% confidence interval)

    return details

def solve_non_linear_regression_error_xy(func, coeff0, x, y, sigma_x = None, sigma_y = None, excluded = None, jacob = None, jacobx = None, options = None, fix_coeff = None):
    ##############################
    # Parameters                 #
    ##############################

    N = len(y)

    if needs_reshaping(N, x) or needs_reshaping(N, y):
        x, y, sigma_x, sigma_y, excluded = reshape_data(N, x, y, sigma_x, sigma_y, excluded)

    if sigma_x is not None:
        if (not hasattr(sigma_x, '__len__') and sigma_x == 0.0) or (hasattr(sigma_x, '__len__') and len(sigma_x) > 1 and len(sigma_x) != N):
            sigma_x = None
        elif not hasattr(sigma_x, '__len__') or len(sigma_x) == 1:
            sigma_x = sigma_x * np.ones(y.shape)

    if sigma_y is not None:
        if (not hasattr(sigma_y, '__len__') and sigma_y == 0.0) or (hasattr(sigma_y, '__len__') and len(sigma_y) > 1 and len(sigma_y) != N):
            sigma_y = None
        elif not hasattr(sigma_y, '__len__') or len(sigma_y) == 1:
            sigma_y = sigma_y * np.ones(y.shape)

    if excluded is None or not hasattr(excluded, '__len__') or len(excluded) != N:
        excluded = np.zeros(N, dtype=bool)

    weighted = True
    method = 'ODR'

    if options is None:
        exclude_outliers = False
        exclude_outliers_num_sigma = 3
        include_statistics = False
    else:
        exclude_outliers = hasattr(options, 'exclude_outliers') and options.exclude_outliers
        exclude_outliers_num_sigma = options.exclude_outliers_num_sigma if hasattr(options, 'exclude_outliers_num_sigma') else 3
        include_statistics = hasattr(options, 'include_statistics') and options.include_statistics
        if hasattr(options, 'method') and options.method != '':
            method = options.method

    ##############################
    # Solve Problem              #
    ##############################

    n = 1
    if exclude_outliers:
        n += 1

    i = 0
    while i < n:
        xFit = x[~excluded]
        yFit = y[~excluded]
        sigma_xFit = sigma_x[~excluded] if sigma_x is not None else None
        sigma_yFit = sigma_y[~excluded] if sigma_y is not None else None

        Nfit = len(yFit)
        df = Nfit - len(coeff0)
        if df <= 0:
            return None

        if not np.any(np.logical_not(fix_coeff)):
            a = np.reshape(coeff0, (len(coeff0),1))
            a_unc = np.zeros((len(coeff0),1))
            C = np.zeros((len(coeff0),len(coeff0)))
            rchi2 = 0.0
        else:
            match method:
                case 'ODR':
                    odr_data = odr.RealData(xFit.flatten(), yFit.flatten(), sx=sigma_xFit.flatten() if sigma_xFit is not None else None, sy=sigma_yFit.flatten() if sigma_yFit is not None else None)
                    odr_fit = odr.ODR(odr_data, odr.Model(func, jacob, jacobx), beta0=coeff0, ifixb=np.logical_not(fix_coeff) if fix_coeff is not None else None)
                    odr_output = odr_fit.run()

                    a = np.reshape(odr_output.beta, (len(odr_output.beta),1))
                    a_unc = np.reshape(odr_output.sd_beta, (len(odr_output.beta),1))
                    C = odr_output.cov_beta
                    non_zero_index = np.argmax(a_unc != 0.0)
                    rchi2 = a_unc[non_zero_index]**2 / C[non_zero_index,non_zero_index]

        if exclude_outliers and i == 0:
            r = yFit - func(a, xFit)
            excluded[~excluded] = is_outlier(r, num_sigma = exclude_outliers_num_sigma).flatten()

        i += 1

    details = FitDetails()
    details.fit_type = method
    details.fit_weighted = weighted
    details.success = not np.any(np.isnan(a))

    details.M = len(a)                                              # Number of Coefficients
    details.N = Nfit                                                # Number of Included Points
    details.NExcl = N - Nfit                                        # Number of Excluded Points
    details.df = df                                                 # Degrees of Freedom

    details.coeff = a                                               # Coefficients
    details.cov = C                                                 # Covariance Matrix
    details.rchi2 = rchi2                                           # Update RChi2 to match that used by ODR

    append_fit_statistics(details, x, y, None, excluded, include_statistics, func, jacob)

    details.coeffVar = np.reshape(np.diag(C), details.coeff.shape)  # Coefficient Variance
    details.coeffUnc = np.sqrt(details.rchi2*details.coeffVar)      # Coefficient Uncertainty (68% confidence interval)

    return details

#endregion

#region Fitting: Specialized

def fit_slope(x, y, sigma = None, excluded = None, options = None):
    N = len(y)

    if needs_reshaping(N, x) or needs_reshaping(N, y):
        x, y, _, sigma, excluded = reshape_data(N, x, y, None, sigma, excluded)

    details = solve_linear_regression(x, y, sigma, excluded, options)

    if details is not None:
        details.fit_type = f"Slope{details.fit_type}"

    return details

def fit_polynomial(degree, x, y, sigma = None, excluded = None, options = None):
    N = len(y)

    if needs_reshaping(N, x) or needs_reshaping(N, y):
        x, y, _, sigma, excluded = reshape_data(N, x, y, None, sigma, excluded)

    if options is None:
        force_use_scaling = False
    else:
        force_use_scaling = hasattr(options, 'force_use_scaling') and options.force_use_scaling

    mi = np.tile(np.arange(degree+1),y.shape)

    use_scaling = force_use_scaling or np.abs(np.log10(np.power(np.max(x), degree))) > 16 # double numeric precision
    if use_scaling:
        x, xMean, xStd = scale_x(x, excluded)

    X = np.power(x, mi)

    details = solve_linear_regression(X, y, sigma, excluded, options)

    if use_scaling:
        unscale_coefficients(details, xMean, xStd)
        details.coeffVar = np.diag(details.cov)
        details.coeffUnc = np.sqrt(details.rchi2*details.coeffVar)

    if details is not None:
        details.fit_type = f"Poly{details.fit_type}"

    return details

def fit_exy(x, y, sigma_x = None, sigma_y = None, excluded = None, options = None):
    if sigma_x is None:
        details = fit_polynomial(1, x, y, sigma_y, excluded, options)
    else:
        details = solve_linear_regression_error_xy(x, y, sigma_x, sigma_y, excluded, options)

    return details

def fit_log_exy(x, y, sigma_x = None, sigma_y = None, excluded = None, options = None, coeff0 = None, fix_coeff = None):
    N = len(y)

    if needs_reshaping(N, x) or needs_reshaping(N, y):
        x, y, sigma_x, sigma_y, excluded = reshape_data(N, x, y, sigma_x, sigma_y, excluded)

    if excluded is None or not hasattr(excluded, '__len__') or len(excluded) != N:
        excluded = np.zeros(N, dtype=bool)

    excluded = excluded | (np.reshape(x, (N,)) <= 0) | (np.reshape(y, (N,)) <= 0)

    log_x = np.zeros(x.shape)
    log_x[~excluded] = np.log10(x[~excluded])
    log_y = np.zeros(y.shape)
    log_y[~excluded] = np.log10(y[~excluded])

    if sigma_x is not None:
        log_sigma_x = np.zeros(sigma_x.shape)
        log_sigma_x[~excluded] = 1/np.log(10) * sigma_x[~excluded] / x[~excluded]
    else:
        log_sigma_x = None

    if sigma_y is not None:
        log_sigma_y = np.zeros(sigma_y.shape)
        log_sigma_y[~excluded] = 1/np.log(10) * sigma_y[~excluded] / y[~excluded]
    else:
        log_sigma_y = None

    if log_sigma_x is None:
        if coeff0 is not None or fix_coeff is not None:
            raise FitException("Not supported")

        details = fit_polynomial(1, log_x, log_y, log_sigma_y, excluded, options)
    else:
        details = solve_linear_regression_error_xy(log_x, log_y, log_sigma_x, log_sigma_y, excluded, options, coeff0 = coeff0, fix_coeff = fix_coeff)

    if details is not None:
        details.fit_type = f"Log{details.fit_type}"

    return details

def power_law_func(coeff, x):
   #return beta[0] *              np.power(       x , beta[1]) # Will throw a warning if x<0 which can happen during optimization
    return coeff[0] * np.sign(x) * np.power(np.abs(x), coeff[1])

def power_law_jacob_coeff(coeff, x):
    beta0Derv = np.power(x, coeff[1])
    beta1Derv = coeff[0] * np.log(x) * np.power(x, coeff[1])

    if np.ndim(coeff) == 2:
        return np.hstack((beta0Derv, beta1Derv))

    return np.vstack((beta0Derv, beta1Derv))

def power_law_jacob_x(coeff, x):
    if coeff[1] == 0.0:
        return np.zeros(x.shape)

    return coeff[0] * coeff[1] * np.power(x, coeff[1] - 1)

def fit_power_law(x, y, sigma_x = None, sigma_y = None, excluded = None, options = None, coeff0 = None, fix_coeff = None):
    N = len(y)

    if needs_reshaping(N, x) or needs_reshaping(N, y):
        x, y, sigma_x, sigma_y, excluded = reshape_data(N, x, y, sigma_x, sigma_y, excluded)

    if excluded is None or not hasattr(excluded, '__len__') or len(excluded) != N:
        excluded = np.zeros(N, dtype=bool)

    excluded = excluded | (np.reshape(x, (N,)) <= 0) | (np.reshape(y, (N,)) <= 0)

    if coeff0 is None:
        fit0_options = deepcopy(options)
        fit0_options.include_statistics = False
        fit0 = fit_log_exy(x, y, sigma_x, sigma_y, excluded, fit0_options, coeff0 = coeff0, fix_coeff = fix_coeff)
        coeff0 = [10**fit0.coeff.flatten()[0], fit0.coeff.flatten()[1]]

    details = solve_non_linear_regression_error_xy(power_law_func, coeff0, x, y, sigma_x, sigma_y, excluded, power_law_jacob_coeff, power_law_jacob_x, options, fix_coeff = fix_coeff)

    if details is not None:
        details.fit_type = f"PL{details.fit_type}"

    return details

#endregion

#region Fitting: Eval Fits

def eval_linear_regression(details, X):
    y, _ = eval_linear_regression_with_uncertainty(details, X)
    return y

def eval_linear_regression_with_uncertainty(details, X):
    if hasattr(details, 'coeff'):
        coeff = details.coeff
    else:
        coeff = details

    y = np.matmul(X, coeff)

    if hasattr(details, 'cov'):
        # yVar = np.reshape(np.diag(np.matmul(np.matmul(X, fit.cov), X.transpose())), y.shape)
        yVar = np.reshape(np.sum(X * np.transpose(np.matmul(np.transpose(details.cov), np.transpose(X))), axis=1), y.shape)

        if hasattr(details, 'rchi2'):
            rchi2 = details.rchi2
        else:
            rchi2 = 1.0

        if rchi2 != 1.0:
            yUnc = np.sqrt(rchi2 * yVar)
    else:
        yUnc = None

    return y, yUnc

def eval_slope(details, x):
    y, _ = eval_slope_with_uncertainty(details, x)
    return y

def eval_slope_with_uncertainty(details, x):
    undo_reshape = False
    if not hasattr(x, '__len__'):
        N = 1
    else:
        N = len(x)

        if np.ndim(x) == 1 or np.shape(x)[0] != N:
            x = np.reshape(x, (N,1))
            undo_reshape = True

    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'i':
            x = np.double(x)
    else:
        if isinstance(x, int):
            x = float(x)

    y, yUnc = eval_linear_regression_with_uncertainty(details, x)

    if N == 1:
        y = y.item(0)
        if yUnc is not None:
            yUnc = yUnc.item(0)
    elif undo_reshape:
        y = np.reshape(y, (N))
        if yUnc is not None:
            yUnc = np.reshape(yUnc, (N))

    return y, yUnc

def eval_polynomial(details, x):
    y, _ = eval_polynomial_with_uncertainty(details, x)
    return y

def eval_polynomial_with_uncertainty(details, x):
    if hasattr(details, 'fit_type'):
        fit_type = details.fit_type
    else:
        fit_type = 'OLS'

    if hasattr(details, 'coeff'):
        coeff = details.coeff
    else:
        coeff = details

    undo_reshape = False
    if not hasattr(x, '__len__'):
        N = 1
    else:
        N = len(x)

        if np.ndim(x) == 1 or np.shape(x)[0] != N:
            x = np.reshape(x, (N,1))
            undo_reshape = True

    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'i':
            x = np.double(x)
    else:
        if isinstance(x, int):
            x = float(x)

    if 'Log' in fit_type:
        x = np.log10(x)

    M = len(coeff)
    mi = np.tile(np.arange(M),(N,1))
    X = np.power(x, mi)

    y, yUnc = eval_linear_regression_with_uncertainty(details, X)

    if N == 1:
        y = y.item(0)
        if yUnc is not None:
            yUnc = yUnc.item(0)
    elif undo_reshape:
        y = np.reshape(y, (N))
        if yUnc is not None:
            yUnc = np.reshape(yUnc, (N))

    if 'Log' in fit_type:
        y = np.power(10, y)
        if yUnc is not None:
            yUnc = np.log(10) * y * yUnc

    return y, yUnc

def eval_power_law(details, x):
    y, _ = eval_power_law_with_uncertainty(details, x)
    return y

def eval_power_law_with_uncertainty(details, x, sigma_x = None):
    if hasattr(details, 'coeff'):
        coeff = details.coeff
    else:
        coeff = details

    undo_reshape = False
    if not hasattr(x, '__len__'):
        N = 1
    else:
        N = len(x)

        if np.ndim(x) == 1 or np.shape(x)[0] != N:
            x = np.reshape(x, (N,1))
            undo_reshape = True

        if sigma_x is not None:
            if np.ndim(sigma_x) == 1 or np.shape(sigma_x)[0] != N:
                sigma_x = np.reshape(sigma_x, (N,1))

    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'i':
            x = np.double(x)
    else:
        if isinstance(x, int):
            x = float(x)

    y =  power_law_func(coeff, x)

    if hasattr(details, 'cov'):
        J = power_law_jacob_coeff(coeff, x)

        # yVar = np.reshape(fit.rchi2 * np.diag(np.matmul(np.matmul(J, fit.cov), J.transpose())), y.shape)
        yVar = np.reshape(details.rchi2 * np.sum(J * np.transpose(np.matmul(np.transpose(details.cov), np.transpose(J))), axis=1), y.shape)

        if hasattr(details, 'rchi2'):
            rchi2 = details.rchi2
        else:
            rchi2 = 1.0

        yUnc = np.sqrt(rchi2 * yVar)
    else:
        yUnc = None

    if sigma_x is not None and yUnc is not None:
        yUnc = np.sqrt(np.power(yUnc, 2) + np.power(power_law_jacob_x(coeff, x) * sigma_x, 2))

    if N == 1:
        y = y.item(0)
        if yUnc is not None:
            yUnc = yUnc.item(0)
    elif undo_reshape:
        y = np.reshape(y, (N))
        if yUnc is not None:
            yUnc = np.reshape(yUnc, (N))

    return y, yUnc

#endregion

#region Fitting: Utilities

def is_outlier(a, num_sigma = 3, method = 'sigma_clip'):
    match method:
        case 'sigma_clip':
            not_nan_filter = ~np.isnan(a)
            if np.any(not_nan_filter):
                outliers = np.ones(a.shape, dtype=np.bool_)
                outliers[not_nan_filter] = sigma_clip(a[not_nan_filter], sigma = num_sigma, maxiters = None).mask
                return outliers
            else:
                return sigma_clip(a, sigma = num_sigma, maxiters = None).mask

        case 'smad':
            if len(a) > 0:
                madfactor = 1.4826
                center = np.median(a)
                amad = madfactor * np.median(np.abs(a - center))
            else:
                center = 0
                amad = 1

            lowerbound = center - num_sigma*amad
            upperbound = center + num_sigma*amad

            return np.logical_or(a > upperbound, a < lowerbound)

    return np.zeros(a.shape, dtype=np.bool_)

def needs_reshaping(N, a):
    return np.ndim(a) == 1 or np.shape(a)[0] != N

def reshape_data(N, x, y, sigma_x = None, sigma_y = None, excluded = None):
    if x is not None:
        if np.ndim(x) == 2:
            if np.shape(x)[0] != N:
                x = np.transpose(x)
        else:
            x = np.reshape(x, (N,1))
    if y is not None:
        y = np.reshape(y, (N,1))
    if sigma_x is not None and hasattr(sigma_x, '__len__') and len(sigma_x) > 1:
        sigma_x = np.reshape(sigma_x, (N,1))
    if sigma_y is not None and hasattr(sigma_y, '__len__') and len(sigma_y) > 1:
        sigma_y = np.reshape(sigma_y, (N,1))
    if excluded is not None and hasattr(excluded, '__len__') and len(excluded) > 1:
        excluded = np.reshape(excluded, (N,))

    return x, y, sigma_x, sigma_y, excluded

def scale_x(x, excluded):
    if excluded is not None and sum(excluded) > 0:
        notExcluded = np.logical_not(excluded)
        xMean = x[notExcluded].mean()
        xStd = x[notExcluded].std()
    else:
        xMean = x.mean()
        xStd = x.std()

    xScaled = (x - xMean) / xStd

    return xScaled, xMean, xStd

def unscale_coefficients(details, xMean, xStd):
    M = len(details.coeff)

    T = np.zeros((M,M))
    for j in range(0,M):
        for k in range(j,M):
            T[j,k] = special.comb(k, k-j) * np.power(-xMean, k-j) / np.power(xStd,k)

    details.scaledXMean = xMean
    details.scaledXStd = xStd
    details.scaledCoeff = details.coeff
    details.scaledCov = details.cov

    details.cov = np.matmul(np.matmul(T, details.cov), np.transpose(T))
    details.coeff = np.matmul(T, details.coeff)

def append_fit_statistics(details, X, y, sigma = None, excluded = None, include_statistics = False, func = None, jacob = None):
    if func is None:
        yModel = np.reshape(np.matmul(X, details.coeff), y.shape)   # Fit Values
    else:
        yModel = func(details.coeff, X)

    yRes = y - yModel                                       # Fit Residuals

    r = yRes[~excluded]                                     # Residuals (included only)

    if not hasattr(details, 'chi2'):
        if not sigma is None:
            sigmaFit = sigma[~excluded]
            details.chi2 = np.sum(np.power(r/sigmaFit, 2))  # Chi Squared
        elif hasattr(details, 'rchi2'):
            details.chi2 = details.rchi2 * details.df
        else:
            sigmaFit = np.std(r)
            details.chi2 = np.sum(np.power(r/sigmaFit, 2))

    if not hasattr(details, 'rchi2'):
        details.rchi2 = details.chi2 / details.df           # Reduced Chi Squared

    if include_statistics:
        sse = np.sum(np.power(r,2))                         # Sum of Squared Estimate of Errors
        details.rms = np.sqrt(sse/details.N)                # Root Mean Square Error

        if not np.any(details.cov == 0.0):
            Dinv = np.diag(1/np.sqrt(np.diag(details.cov)))
            details.correlation = np.matmul(np.matmul(Dinv, details.cov), Dinv) # Correlation Matrix
        else:
            details.correlation = None

        if jacob is None:
            J = X
        else:
            J = jacob(details.coeff, X)

        # yModelVar = np.reshape(fit.rchi2 * np.diag(np.matmul(np.matmul(J, fit.cov), J.transpose())), y.shape)
        yModelVar = np.reshape(details.rchi2 * np.sum(J * np.transpose(np.matmul(np.transpose(details.cov), np.transpose(J))), axis=1), y.shape)

        details.fit = yModel
        details.fitUnc = np.sqrt(yModelVar)
        details.residual = yRes                             # Fit Residuals
        details.excluded = excluded                         # Excluded Points

        critical = -stats.t.ppf((1-0.68)/2, details.df)     # 68% Confidence Interval

        confidenceDelta = critical * np.sqrt(sse/details.df + yModelVar)
        details.fitConfidenceUpper = yModel + confidenceDelta
        details.fitConfidenceLower = yModel - confidenceDelta

        predictionDelta = critical * details.fitUnc
        details.fitPredictionUpper = yModel + predictionDelta
        details.fitPredictionLower = yModel - predictionDelta

        yMean = np.mean(y[~excluded])                       # Mean y (included only)
        sst = np.sum(np.power(y[~excluded]-yMean,2))        # Total Sum of Squares
        details.R2 = 1 - sse / sst                          # Pearson Correlation Coefficient

def simplify_fit(details):
    simpleDetails = FitDetails()
    simpleDetails.fit_type = details.fit_type
    simpleDetails.N = details.N
    simpleDetails.coeff = details.coeff.copy()
    simpleDetails.coeffUnc = details.coeffUnc.copy()
    simpleDetails.cov = details.cov.copy()
    simpleDetails.rchi2 = details.rchi2
    return simpleDetails

def reset_simplified_fit(details):
    emptyDetails = deepcopy(details)
    emptyDetails.N = 0
    emptyDetails.coeff[:,:] = 0.0
    emptyDetails.coeffUnc[:,:] = 0.0
    emptyDetails.cov[:,:] = 0.0
    emptyDetails.rchi2 = 0.0
    return emptyDetails

def get_fit_function_text(details, x_label, y_label):
    if len(details.coeff) > 2:
        raise FitException('Not supported')

    if 'PL' in details.fit_type:
        if details.coeffUnc[0] > 0.0 and details.coeffUnc[1] > 0.0:
            return f"${y_label} = ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})\\:{x_label}^{{({details.coeff[1,0]:.2f}\\pm{details.coeffUnc[1,0]:.2f})}}$"

        if details.coeffUnc[0] > 0.0:
            if len(details.coeff) == 1 or details.coeff[1] == 1.0:
                return f"${y_label} = ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})\\:{x_label}$"

            return f"${y_label} = ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})\\:{x_label}^{details.coeff[1,0]:.2f}$"

        if len(details.coeff) == 1 or details.coeff[1] == 1.0:
            return f"${y_label} = {details.coeff[0,0]:.2f}\\:{x_label}$"

        return f"${y_label} = {details.coeff[0,0]:.2f}\\:{x_label}^{details.coeff[1,0]:.2f}$"

    if 'Log' in details.fit_type:
        if details.coeffUnc[0] > 0.0 and details.coeffUnc[1] > 0.0:
            return f"$\\log {y_label} = ({details.coeff[1,0]:.2f}\\pm{details.coeffUnc[1,0]:.2f})\\:\\log {x_label} + ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})$"

        if details.coeffUnc[0] > 0.0:
            return f"$\\log {y_label} = {details.coeff[1,0]:.2f}\\:\\log {x_label} + ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})$"

        if details.coeffUnc[1] > 0.0:
            return f"$\\log {y_label} = ({details.coeff[1,0]:.2f}\\pm{details.coeffUnc[1,0]:.2f})\\:\\log {x_label} + {details.coeff[0,0]:.2f}$"

        return f"$\\log {y_label} = {details.coeff[1,0]:.2f}\\:\\log {x_label} + {details.coeff[0,0]:.2f}$"

    if 'Slope' in details.fit_type:
        if details.coeffUnc[0] > 0.0:
            return f"${y_label} = ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})\\:{x_label}$"

        return f"${y_label} = {details.coeff[0,0]:.2f}\\:{x_label}$"

    if 'Poly' in details.fit_type:
        if details.coeffUnc[0] > 0.0 and details.coeffUnc[1] > 0.0:
            return f"${y_label} = ({details.coeff[1,0]:.2f}\\pm{details.coeffUnc[1,0]:.2f})\\:{x_label} + ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})$"

        if details.coeffUnc[0] > 0.0:
            return f"${y_label} = {details.coeff[1,0]:.2f}\\:{x_label} + ({details.coeff[0,0]:.2f}\\pm{details.coeffUnc[0,0]:.2f})$"

        if details.coeffUnc[1] > 0.0:
            return f"${y_label} = ({details.coeff[1,0]:.2f}\\pm{details.coeffUnc[1,0]:.2f})\\:{x_label} + {details.coeff[0,0]:.2f}$"

        return f"${y_label} = {details.coeff[1,0]:.2f}\\:{x_label} + {details.coeff[0,0]:.2f}$"

    raise FitException('Not supported')

#endregion
