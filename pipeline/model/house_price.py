# import created modules
#from pipeline.model.source.data_cleaning import DataCleaning as dtcln
#from source.data_cleaning import describe_with_tukey_fences
#import source.modeling as modeling
#import source.evaluation as evaluation
from pipeline.model.source.data_cleaning import DataCleaning as dtcln
from pipeline.model.source.data_cleaning import describe_with_tukey_fences
import pipeline.model.source.modeling as modeling
import pipeline.model.source.evaluation as evaluation

# import standard libraries
import os
import numpy as np
import pickle
import pandas as pd
import scipy.stats as stats

# import scikit modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from tabulate import tabulate
# below two packages are for feature selection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import pipeline.model.data as data
import pipeline.model.assets.outputs as outputs
REAL_ESTATE_CSV_FILEPATH = os.path.join(os.getcwd(), 'data', 'clean_dataset.csv')
CLEANED_CSV_FILEPATH = os.path.join(os.getcwd(), 'assets','outputs', 'df_after_cleaning.csv')

# ['source','land_surface', 'facades_number', 'swimming_pool_has','postcode_median_price',
#              'property_subtype_median_facades_number', 'building_state_agg_median_price']
NUM_CV_FOLDS = 3
DEGREE_MAX = 3

cleaner = dtcln(csv_filepath=REAL_ESTATE_CSV_FILEPATH)
df, df_outliers = cleaner.get_cleaned_dataframe(cleaned_csv_path=CLEANED_CSV_FILEPATH)
# excluding text columns as requested (boolean kept)
df = df.select_dtypes(exclude=['object'])
df = df.drop(['source', 'land_surface', 'swimming_pool_has'], axis=1)
# calculating price per metre square to detect outliers
df['price_per_sqmtr'] = df['price'] / df['area']
postcode_stats = df['postcode'].value_counts(ascending=False)
postcode_value_less_than_10 = postcode_stats[postcode_stats <= 10]
postcode_value_less_than_10
df['postcode'] = df['postcode'].apply(lambda x: '9999' if x in postcode_value_less_than_10 else x)


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('postcode'):
        m = np.mean(subdf.price_per_sqmtr)
        sd = np.std(subdf.price_per_sqmtr)
        reduced_df = subdf[(subdf.price_per_sqmtr > (m - (2 * sd))) & (subdf.price_per_sqmtr <= (m + (2 * sd)))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


# Applying the function on our dataframe
df = remove_pps_outliers(df)
# Now, we can drop price per metre square column as our outlier detection is done '''
df = df.drop(['price_per_sqmtr'], axis='columns')


class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p < alpha:
            result = "{0} is IMPORTANT for Prediction".format(colX)
        else:
            result = "{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        # print(result)

    def TestIndependence(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        self._print_chisquare_result(colX, alpha)


# Initialize ChiSquare Class
cT = ChiSquare(df)

# Feature Selection
testColumns = ['postcode', 'house_is', 'rooms_number',  # 'property_subtype'
               'area', 'equipped_kitchen_has', 'furnished', 'open_fire', 'terrace',
               'garden']  # , 'building_state_agg'] #'region',
for var in testColumns:
    cT.TestIndependence(colX=var, colY="price")

# Drop the features which are irrelevant as per chi-square '''
df = df.drop(['furnished', 'garden'], axis=1)  # 'property_subtype'
dummies = pd.get_dummies(df, prefix='', prefix_sep='')
X = df.drop(['price'], axis='columns')

y = df.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
lin_reg = modeling.OLS_linear_regression(X_train, y_train)
# Plots learning curves (R² Score) based on training data and k-folds cross-validation
modeling.plot_OLS_lin_reg_r2_curves(X_train, y_train, num_cv_folds=NUM_CV_FOLDS)
# Plots learning curves (Mean Squared Error) based on training data and k-folds cross-validation
modeling.plot_OLS_lin_reg_MSE_curves(X_train, y_train, num_cv_folds=NUM_CV_FOLDS)
X.to_csv('X_data.csv')

def predict_price(model, prop_type, postcode, area, rooms, garden, terrace, prop_condition):
    """
    Function predicts the price with random input data.


    :param prop_type: property type data
    :param postcode: postcode data
    :param area: area
    :param rooms: number of rooms
    :param garden: garden
    :param terrace: terrace
    :param prop_condition: proeprty condition
    :return: price value
    """
    loc_index = np.where(X.columns == postcode)[0]
    prop_condition_index = np.where(X.columns == prop_condition)[0]
    x = np.zeros(len(X.columns))
    x[0] = prop_type
    x[1] = rooms
    x[2] = area
    x[3] = terrace
    x[4] = garden

    if loc_index >= 0:
        x[loc_index] = 1

    if prop_condition_index >= 0:
        x[prop_condition_index] = 1

    return model.predict([x])[0]


model_evaluation_obj = evaluation.Model_Evaluation(lin_reg)
ytrain_predictions, ytest_predictions = model_evaluation_obj.get_predictions(X_train, X_test)
model_evaluation_obj.predict_model(X_train, y_train, X_test, y_test)
predicted_price = predict_price(lin_reg, 1, 8300, 100, 3, 1, 0, 'good')
print(predicted_price)
pickle.dump(lin_reg, open('houseprice_model.pkl', 'wb'))

import json
json_data=X['area'].to_json()

with open('House_Prediction.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)
columns = {'data_columns': [col.lower() for col in X.columns]
           }
with open('data_columns.json', 'w') as f:
    f.write(json.dumps(columns))



#with open("data_columns.json", "r") as f:
    #data_cols = json.load(f)['data_columns']
    #print('datacolssssss',data_cols)

"""    
def linearc_regression_to_json(lrmodel, file=None):
    if file is not None:
        serialize = lambda x: json.dumps(x)
    else:
        serialize = json.dumps
    data = {}
    data['init_params'] = lrmodel.get_params()
    data['model_params'] = mp = {}
    for p in ('coef_', 'intercept_'):
        mp[p] = getattr(lrmodel, p).tolist()
    return serialize(data)

def linear_regression_from_json(jstring):
    data = json.loads(jstring)
    model = LinearRegression(**data['init_params'])
    for name, p in data['model_params'].items():
        setattr(model, name, np.array(p))
    return model
a=linearc_regression_to_json(lin_reg,json_data)
b=linear_regression_from_json(a)
pickle.dump(b,open('test_model.pkl','wb'))
pickle.load(open('test_model.pkl','rb'))
"""

data_cols = None
model = None
postcodes = None


def predict_price(prop_type, postcode, area, rooms, garden, terrace, prop_condition):
    """
    Function predicts the price with random input data.


    :param prop_type: property type data
    :param postcode: postcode data
    :param area: area
    :param rooms: number of rooms
    :param garden: garden
    :param terrace: terrace
    :param prop_condition: proeprty condition
    :return: price value
    """

    loc_index = np.where(data_cols.columns == postcode)[0]
    prop_condition_index = np.where(data_cols.columns == prop_condition)[0]
    x = np.zeros(len(X.columns))
    x[0] = prop_type
    x[1] = rooms
    x[2] = area
    x[3] = terrace
    x[4] = garden

    if loc_index >= 0:
        x[loc_index] = 1

    if prop_condition_index >= 0:
        x[prop_condition_index] = 1

    return model.predict([x])[0]


def load_json_data():
    global data_cols
    global postcodes

    with open("data_columns.json", "r") as f:
        data_cols = json.load(f)['data_columns']
        print(data_cols)
        postcodes = data_cols[1:]

    global model
    if model is None:
        with open("House_Prediction.pkl", "rb") as f:
            model = pickle.load(f)


def get_postcodes():
    return postcodes


def get_data_columns():
    return data_cols


if __name__ == '__main__':
    load_json_data()