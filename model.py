from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_regression_model(data):
    X = data.drop(['amount', 'sick_on_mon_more_six'], axis=1)
    y = data['amount']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler


def train_classification_model(data):
    X = data.drop(['amount', 'sick_on_mon_more_six'], axis=1)
    y = data['sick_on_mon_more_six']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model
