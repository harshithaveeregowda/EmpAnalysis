import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    #data = pd.read_csv(filepath)
    data = pd.read_excel("data1.xlsx")
    return data


def preprocess_data(data):
    data['sick_on_mon_more_six'] = data['sick_on_mon_more_six'].map({'no': 0, 'yes': 1})

    categorical_cols = ['grade', 'party_hometown_on_sun', 'bridge_days_on_tue', 'city_type']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    data = data.drop(columns=['name', 'year'])

    return data
