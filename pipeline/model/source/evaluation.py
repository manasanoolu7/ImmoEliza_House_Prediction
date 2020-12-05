import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Model_Evaluation:

    def __init__(self, model):

        self.model = model

    def get_predictions(self, train_data, test_data):
        """
        Function provides predicted values from train data and test data.

        :param test_data: an array that contains test data
        :param train_data: an array that contains train data

        :return: array of predicted train and test data
        """

        train_predictions = self.model.predict(train_data)
        test_predictions = self.model.predict(test_data)

        return train_predictions, test_predictions

    def predict_model(self, X_train, y_train, X_test, y_test):
        """
        Function generates evaluation data on original inputs and predicted data.
        :param X_train: an array that contains training data to be preticted
        :param y_train: an array that contains training target data
        :param X_test: an array that contains testing data to be predicted
        :param y_test: an array that contains testing target data
        """

        ytrain_predictions, ytest_predictions = self.get_predictions(X_train, X_test)

        mae = mean_absolute_error(y_test, ytest_predictions)

        mse = mean_squared_error(y_test, ytest_predictions)
        rmse = np.sqrt(mse)
        mape = (mse/y_test)*100
        train_score = r2_score(y_train, ytrain_predictions)
        test_score = r2_score(y_test, ytest_predictions)

        Results = pd.DataFrame({'MAE': mae, 'MSE': mse, 'RMSE': rmse,
                                'Train_RSquare': train_score, 'Test_RSquare': test_score}, index=['Values'])
        print('Evaluation Metrics')
        print(tabulate(Results, headers='keys', tablefmt='fancy_grid'))

        plt.subplot(1, 2, 1)
        # display plots
        plt.title('Actual vs Predicted Data\n')
        plt.scatter(y_test, ytest_predictions, c=np.arange(len(y_test)),cmap='Blues')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.subplot(1, 2, 2)
        plt.title('Residuals Distribution Plot\n')
        # histogram of the residuals. It tells how well the residuals are distributed from proposed model
        sns.distplot(y_test - ytest_predictions)
        plt.tight_layout()
        plt.show()