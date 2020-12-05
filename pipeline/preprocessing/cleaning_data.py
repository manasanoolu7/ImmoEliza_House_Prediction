# Import necessary libraries

import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import missingno as msno
import re

url = "https://raw.githubusercontent.com/adamflasse/dataCleaning/main/updated_1.csv"
df = pd.read_csv(url)


# Define functions for data preprocessing tasks


def general_clean(df):
    """Remove duplicates and unnecessary columns"""

    # Drop duplicates
    df.drop_duplicates(subset=['price', 'area', 'rooms_number'], inplace=True)

    # Delete postcode and house_is
    df.drop(
        [
            "postcode",
            "house_is",
            "region",
            'building_state',
            'swimming_pool_has'
        ],
        axis=1,
        inplace=True,
    )

    """Handling with Missing values"""

    # Replace Not Specified by np.nan
    df.replace("Not specified", np.nan, inplace=True)

    """Replace True and False by numerical values"""

    df.replace([True, "True", False, "False"], [1, 1, 0, 0], inplace=True)

    """Cleaning PRICE column: removing anomalies and outliers"""

    # Delete price rows with anomalies (extreme, mistaken values, e.g. 123456789)
    df.drop(df[df["price"] > 20000000].index, inplace=True)
    df.drop(df[df["price"] == 12345678].index, inplace=True)

    # Remove price outliers
    index_price = df[(df["price"] < 10000)].index
    df.drop(index_price, inplace=True)

    # Remove anomalies based on price and area features
    index_area_price = df[(df["area"] > 1000) & (df["price"] < 200000)].index
    df.drop(index_area_price, inplace=True)

    # Remove anomalies based on price and room features
    index_rooms_price = df[(df["rooms_number"] < 4) & (df["price"] > 1000000)].index
    df.drop(index_rooms_price, inplace=True)

    """Cleaning AREA column"""

    # Remove area outliers
    index_area = df[(df["area"] < 5)].index
    df.drop(index_area, inplace=True)

    """Cleaning ROOMS_NUMBER column"""

    # Replace room_number values that are == to area by NaN
    df.loc[(df.rooms_number == df.area), "rooms_number"] = np.nan

    """Cleaning PROPERTY_SUBTYPE column"""

    # Formatting values
    df["property_subtype"] = df["property_subtype"].str.lower()
    df["property_subtype"].replace(to_replace="-", value="_", regex=True, inplace=True)

    "Reduce property_subtype groups"

    # Reduce number of property_subtypes by grouping lower frequencied ones into "other" category
    # ("ground_floor" acted as threshold). This is done to reduce noise in the model
    df["property_subtype"] = df["property_subtype"].replace(
        [
            "exceptional_property",
            "flat_studio",
            "mansion",
            "town_house",
            "loft",
            "country_cottage",
            "service_flat",
            "bungalow",
            "farmhouse",
            "triplex",
            "other_property",
            "manor_house",
            "chalet",
            "castle",
            "kot",
            "penthouse",
            "duplex",
            "mixed_use_building",
            "villa",
            "apartment_block",
            "ground_floor"
        ],
        "OTHERS",
    )
    df["property_subtype"].replace({"apartment": "APARTMENT", "house": "HOUSE"}, inplace=True)

    # change column names with proper ones

    df = df.rename(
        columns={
            "property_subtype": "property-type",
            "kitchen_has": "equipped-kitchen",
            "facades_number": "facades-number",
            "rooms_number": "rooms-number",
        }
    )

    df.columns = df.columns.str.replace(' ', '')

    """Leave dataframe ready for next step"""
    # Reset index after dropping
    df = df.reset_index(drop=True)
    return df


def remove_na_all(df):
    # Remove all observations containing missing values
    df.dropna(axis=0, inplace=True)
    return df


def remove_nas_above30perc(df):
    # Remove observations containing more than 30% of missing values.
    # Missing values will be imputed
    df.dropna(axis=0, thresh=8, inplace=True)
    return df


def preprocessing(df):
    # Use sklearn libraries to create dummy variables to make one-hot encoder for our categorical values
    from sklearn.preprocessing import OneHotEncoder
    import category_encoders as ce

    ohe = ce.OneHotEncoder(handle_unknown="ignore", use_cat_names=True)
    df_ohe = ohe.fit_transform(df)
    return df_ohe


df = general_clean(df)
df = remove_nas_above30perc(df)
df = preprocessing(df)

# Impute missing values with means (remember that we have no observations with more than 2 missing values)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.NaN, strategy="mean")

df["terrace"] = imputer.fit_transform(df["terrace"].values.reshape(-1, 1))[:, 0]
df["equipped-kitchen"] = imputer.fit_transform(
    df["equipped-kitchen"].values.reshape(-1, 1)
)[:, 0]
df["rooms-number"] = imputer.fit_transform(df["rooms-number"].values.reshape(-1, 1))[
                     :, 0
                     ]
df["garden"] = imputer.fit_transform(df["garden"].values.reshape(-1, 1))[:, 0]
df["furnished"] = imputer.fit_transform(df["furnished"].values.reshape(-1, 1))[:, 0]

df.info()

processed_csv = df.to_csv("ready_to_model_df.csv")
