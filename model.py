import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline

df = pd.read_csv('data/tips.csv')

target = 'tip'
y = df[target]
X = df.drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    (['total_bill'], StandardScaler()),
    ('sex', LabelBinarizer()),
    ('smoker', LabelBinarizer()),
    ('day', LabelEncoder()),
    ('time', LabelEncoder()),
    (['size'], StandardScaler())],df_out=True)


Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

model = Lasso(alpha=0.1)
model.fit(Z_train,y_train)
model.score(Z_train,y_train)
model.score(Z_test,y_test)

pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))

X_train.sample().to_dict(orient = 'list')

new = pd.DataFrame({
'total_bill': [11.87],
 'sex': ['Female'],
 'smoker': ['No'],
 'day': ['Thur'],
 'time': ['Lunch'],
 'size': [2]})

type(pipe.predict(new)[0])

prediction = float(pipe.predict(new)[0])
type(round(prediction, 2))
