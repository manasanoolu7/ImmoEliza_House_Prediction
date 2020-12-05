import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler, MinMaxScaler

def OLS_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Returns a linear regression model fitted with Ordinary Least Squares method
    :param X_train: training data (features)
    :param y_train: training target
    :return the fitted model
    """

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    return lin_reg

def OLS_linear_regression_scaled(X_train: np.ndarray, y_train: np.ndarray, scaling: str) -> Pipeline:
    """
    Returns a linear regression model fitted with Ordinary Least Squares method after a particular scaling
    :param X_train: training data (features)
    :param y_train: training target
    :param scaling: type of scaler to be used
    :return the fitted model
    """
    if scaling == 'std':
        reg = make_pipeline(StandardScaler(), LinearRegression())
    elif scaling == 'robust':
        reg = make_pipeline(RobustScaler(), LinearRegression())
    elif scaling == 'minmax':
        reg = make_pipeline(MinMaxScaler(), LinearRegression())
    else: #no scaling
        reg = make_pipeline(LinearRegression())
    reg.fit(X_train, y_train)
    return reg

def plot_OLS_lin_reg_r2_curves(X_train: np.ndarray, y_train: np.ndarray, num_cv_folds: int) -> None:
    """
    Plots learning curves (R² score) based on training data and k-folds cross-validation
    :param X_train: training data (features)
    :param y_train: training target
    :param num_cv_folds: number of folds for the cross-validation of the model
    """

    train_sizes_abs, train_r2_scores, valid_r2_scores = learning_curve(
        LinearRegression(), X_train, y_train, scoring = 'r2',
        train_sizes=np.linspace(0.1, 1.0, 10), cv=num_cv_folds)
    
    train_r2_scores_mean = np.mean(train_r2_scores, axis=1)
    train_r2_scores_std = np.std(train_r2_scores, axis=1)
    valid_r2_scores_mean = np.mean(valid_r2_scores, axis=1)
    valid_r2_scores_std = np.std(valid_r2_scores, axis=1)

    plt.plot(train_sizes_abs, train_r2_scores_mean, color="darkorange",
        marker='o', linestyle='--', label= 'Training score')
    plt.fill_between(train_sizes_abs, train_r2_scores_mean - train_r2_scores_std,
        train_r2_scores_mean + train_r2_scores_std, alpha=0.2, color="darkorange")
    plt.plot(train_sizes_abs, valid_r2_scores_mean, color="navy",
        marker='o', linestyle='--', label='Cross-validation score')
    plt.fill_between(train_sizes_abs, valid_r2_scores_mean - valid_r2_scores_std,
        valid_r2_scores_mean + valid_r2_scores_std, alpha=0.2, color="navy")
    plt.xticks(train_sizes_abs)
    plt.xlabel('# Training examples')
    plt.ylabel('R² Score')
    plt.legend(loc='best')
    plt.title(f'Learning Curves\n{num_cv_folds}-fold Cross-Validation')

    plt.show()

def plot_OLS_lin_reg_MSE_curves(X_train: np.ndarray, y_train: np.ndarray, num_cv_folds: int) -> None:
    """
    Plots learning curves (Mean Squared Error) based on training data and k-folds cross-validation
    :param X_train: training data (features)
    :param y_train: training target
    :param num_cv_folds: number of folds for the cross validation
    """

    train_sizes_abs, train_negMSE_scores, valid_negMSE_scores = learning_curve(
        LinearRegression(), X_train, y_train, scoring = 'neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), cv=num_cv_folds)
    
    train_MSE_mean = np.mean(-train_negMSE_scores, axis=1)
    train_MSE_std = np.std(-train_negMSE_scores, axis=1)
    valid_MSE_mean = np.mean(-valid_negMSE_scores, axis=1)
    valid_MSE_std = np.std(-valid_negMSE_scores, axis=1)

    plt.plot(train_sizes_abs, train_MSE_mean, color="darkorange",
        marker='o', linestyle='--', label= 'Training score')
    plt.fill_between(train_sizes_abs, train_MSE_mean - train_MSE_std,
        train_MSE_mean + train_MSE_std, alpha=0.2, color="darkorange")
    plt.plot(train_sizes_abs, valid_MSE_mean, color="navy",
        marker='o', linestyle='--', label='Cross-validation score')
    plt.fill_between(train_sizes_abs, valid_MSE_mean - valid_MSE_std,
        valid_MSE_mean + valid_MSE_std, alpha=0.2, color="navy")
    plt.xticks(train_sizes_abs)
    plt.xlabel('# Training examples')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.title(f'Learning Curves\n{num_cv_folds}-fold Cross-Validation')

    plt.show()

def plot_poly_reg_validation_curves(X_train: np.ndarray, y_train: np.ndarray, num_cv_folds: int, degree_max: int) -> None:
    """
    Plots validation curves (R² score) based on training data and k-folds cross-validation for different
    degrees of Polynomial Regression
    :param X_train: training data (features)
    :param y_train: training target
    :param num_cv_folds: number of folds for the cross validation
    :param degree_max: maximal degree of the polynomial models to validate
    """

    degree_range = np.arange(1, degree_max+1)

    train_scores, valid_scores = validation_curve(make_pipeline(PolynomialFeatures(), LinearRegression()),
        X_train, y_train, param_name='polynomialfeatures__degree', param_range=degree_range, cv=num_cv_folds)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    plt.plot(degree_range, train_scores_mean, color="darkred", marker='o', linestyle='--', label= 'Training score')
    plt.fill_between(degree_range, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.2, color="darkred")
    plt.plot(degree_range, valid_scores_mean, color="seagreen",   marker='o', linestyle='--', label='Cross-validation score')
    plt.fill_between(degree_range, valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std, alpha=0.2, color="seagreen")
    plt.xticks(degree_range)
    plt.xlabel('Polynomial Regression Degree')
    plt.ylabel('R² Score')
    plt.legend(loc='best')
    plt.title(f'Validation Curve with Polynomial Regression\n{num_cv_folds}-fold Cross-Validation')
        
    plt.show()