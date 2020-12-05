"""
IMPORTANT DISCLAIMER
File intended only for internal(private) use by the team (not so cleaned code).
No extended documentation provided since csv output available in data.


"""

import pandas as pd
import numpy as np
import os
from typing import List
from typing import Dict

#visualisation for testing purpose
pd.options.display.max_rows = 20
pd.options.display.max_columns = None

REAL_ESTATE_CSV_FILEPATH = os.path.join(os.path.dirname(os.getcwd()), "data", "clean_dataset.csv")
STATBEL_REAL_ESTATE_CSV_FILEPATH = os.path.join(os.path.dirname(os.getcwd()), "data", "statbel_date_facades_price_median_and_selling_by_municipality.csv")
# Features to properly read following file not implemented yet
# Objectives were to:
# 1) obtain a possible more precise feature than postcode median price, i.e. median price by postcode, surface and/or building type
# 2) if not abovementioned feature not used, still could be compared with predicted modelled results
STATBEL_REAL_ESTATE_BY_SURFACE_CSV_FILEPATH = os.path.join(os.path.dirname(os.getcwd()), "data", "statbel date facades surface_range price_median and selling by municipality.csv")
MUNICIPALITIES_CSV_FILEPATH = os.path.join(os.path.dirname(os.getcwd()), "data", "zipcode-belgium.csv")
CLEANED_CSV_FILEPATH = os.path.join(os.path.dirname(os.getcwd()), "assets", "outputs", "df_with_statbel.csv")
SURFACE_COLUMN = "Superficie du terrain"
SURFACE_RANGE_EXCLUDED = "Superficie inconnue"
SURFACE_RANGE = {"0-99m²":[0,99], "100-299m²":[100,299], "300-599m²":[300,599],
                 "600-999m²":[600,999], "1000-1499m²":[1000,1499], ">=1500m²":[1500,9999]}

#paths for windows users
REAL_ESTATE_CSV_FILEPATH_WIN = os.path.dirname(os.getcwd()) + r"\data" + "\clean_dataset.csv"
STATBEL_REAL_ESTATE_CSV_FILEPATH_WIN = os.path.dirname(os.getcwd()) + r"\data" + "\statbel date facades price_median and selling by municipality.csv"
STATBEL_REAL_ESTATE_BY_SURFACE_CSV_FILEPATH_WIN = os.path.dirname(os.getcwd()) + r"\data" + "\statbel date facades surface_range price_median and selling by municipality.csv"
MUNICIPALITIES_CSV_FILEPATH_WIN = os.path.dirname(os.getcwd()) + r"\data" + "\zipcode-belgium.csv"
CLEANED_CSV_FILEPATH_WIN = os.path.dirname(os.getcwd()) + r"\assets" + r"\outputs" + "df_after_cleaning.csv"

def add_postcode_price(statbel_csv_filepath: str = STATBEL_REAL_ESTATE_CSV_FILEPATH,
             municipalities_csv_filepath: str = MUNICIPALITIES_CSV_FILEPATH,
             real_estate_csv_filepath: str = REAL_ESTATE_CSV_FILEPATH,
             building_type: str = None):
    """
    Function adding postcode price (plus latitude and longitude) to the original dataframe by combing with other datasets
    :param statbel_csv_filepath: filepath to the extracted official real estate statistics
    :param municipalities_csv_filepath: filepath to the file matching municipalities and postcodes
    :param real_estate_csv_filepath: filepath to the scrapped real estate csv file
    :param building_type: building type to be filtered out

    :return:
    """
    if (statbel_csv_filepath == STATBEL_REAL_ESTATE_BY_SURFACE_CSV_FILEPATH or
        statbel_csv_filepath == STATBEL_REAL_ESTATE_BY_SURFACE_CSV_FILEPATH_WIN):
        surface_column = True

    real_estate_0 = pd.read_csv(real_estate_csv_filepath)
    real_estate_out = real_estate_0.copy(deep=True)
    re_postcodes = list(real_estate_out.loc[:,"postcode"].unique())

    municipalities = pd.read_csv(municipalities_csv_filepath, header=None)
    municipalities_columns = ['postcode', 'municipality', 'longitude', 'latitude']
    municipalities.rename(columns=dict(zip(municipalities.columns, municipalities_columns)), inplace=True)
    municipalities = municipalities[municipalities.postcode.isin(re_postcodes)]

    delimiter = ','
    if surface_column:
        delimiter = ';'
    statbel = pd.read_csv(statbel_csv_filepath, delimiter=delimiter)
    #fill na to avoid zeros later
    statbel.fillna(0, inplace=True)

    statbel_columns = ["region_sb", "province", "district", "municipality", "year", "building_type", "price_median",
                       "sellings"]
    if surface_column:
        statbel_columns = ["province", "district", "municipality", "year", "building_type", "surface_range",
                           "price_median", "sellings"]

    statbel.rename(columns=dict(zip(statbel.columns, statbel_columns)), inplace=True)

    if building_type is not None:
        statbel =  statbel[statbel.building_type == building_type]

    if surface_column:
        statbel = statbel[statbel.surface_range != SURFACE_RANGE_EXCLUDED]

    statbel["price_median"] = pd.to_numeric(statbel["price_median"], errors = 'coerce')
    statbel.dropna(subset=['price_median', 'sellings']) #added on 25/11/20

    statbel["pm_per_s"] = statbel.apply(lambda x: x["price_median"] * x["sellings"], axis=1)
           
    # first matching all in dataset, all detected but Antwerp is both 2000 and 2060
    #applying statbel only on selected postcode. Note: a few locations have no province in official statistics.
    statbel_pc = municipalities.merge(statbel, on='municipality', how='left')
    #saving complete file before manipulation

    by_columns = ["postcode"]
    if surface_column:
        by_columns = [["postcode", "surface_range"]]

    sb_postcode = statbel_pc.loc[:, ['postcode', 'pm_per_s', 'sellings', 'latitude', 'longitude']].groupby(by=by_columns).agg(
        {'pm_per_s': sum, 'sellings':sum, 'latitude':np.median, 'longitude': np.median})
    sb_postcode["price_m_by_postcode"] = sb_postcode["pm_per_s"] / sb_postcode["sellings"]
    sb_postcode.loc[:, ["price_m_by_postcode", 'latitude', 'longitude']] = sb_postcode.loc[:,
        ["price_m_by_postcode", 'latitude', 'longitude']].apply(lambda x: round(x, 2))

    #drop longitude and latitude before getting median values by postcode
    statbel_pc.drop(columns=["longitude", "latitude"], inplace=True)
    statbel_pc = statbel_pc.merge(sb_postcode.loc[:, ["price_m_by_postcode", "longitude", "latitude"]],
                                  on=["postcode"], how='left')
    #statbel_pc.rename(columns={"price_m_by_postcode": "price_m_by_postcode"}, inplace=True)

    #then I can remove municipalities keeping only valid postcode

    # filtering valid values before removing duplicates after
    statbel_pc = statbel_pc[statbel_pc.pm_per_s > 0]
    # after grouping only location-related columns are kept in statbel_pc before merging with results
    statbel_pc = statbel_pc.drop(columns=["municipality", "year", "building_type", "price_median", "sellings", "pm_per_s"])
    statbel_pc.drop_duplicates(inplace=True)

    out_columns = ["postcode", "price_m_by_postcode", "latitude", "longitude" ] # "price_m_by_district", "price_m_by_province", "price_m_by_region"]
    if surface_column:
        out_columns += "surface_range"

    real_estate_out = real_estate_out.merge(statbel_pc.loc[:, out_columns], on='postcode', how='left')

    return real_estate_out







#TESTING ON WINDOWS (to exclude as comment when running Jupyter NB)
"""
df = add_postcode_price(statbel_csv_filepath= STATBEL_REAL_ESTATE_BY_SURFACE_CSV_FILEPATH_WIN, #STATBEL_REAL_ESTATE_CSV_FILEPATH_WIN,
                        municipalities_csv_filepath = MUNICIPALITIES_CSV_FILEPATH_WIN,
                        real_estate_csv_filepath = REAL_ESTATE_CSV_FILEPATH_WIN,
                        )

df.to_csv(CLEANED_CSV_FILEPATH)
"""
#INTERNAL CONSIDERATION
#Index price overs year could have been better but influence expected to be not so high anyway

