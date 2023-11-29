from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import BytesIO
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
app = FastAPI()

df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
df_train = df_train.drop_duplicates(subset=[x for x in df_train.columns if x != "selling_price"])
weird_features = ["mileage", "engine", "max_power"]
for name in weird_features:
    df_train[name]=df_train[name].str.split().str[0]
df_train.drop("torque", axis = 1, inplace = True)
for name in weird_features:
    df_train[name]=pd.to_numeric(df_train[name], errors = "coerce")
columns = ["mileage", "engine", "max_power", "seats"]
for col in columns:
    df_train[col] = df_train[col].fillna(df_train[col].median())
df_train["engine"]=df_train["engine"].astype(int)
df_train["seats"]=df_train["seats"].astype(int)

y_train = df_train["selling_price"]
X_train_cat =df_train.drop(["selling_price", "name"], axis = 1, inplace = False)
X_traincat_new= X_train_cat.copy()
X_traincat_new["square of year"] =X_traincat_new["year"]*X_traincat_new["year"]
X_traincat_new = X_traincat_new.drop(labels ="year",axis = 1)
X_traincat_new['fuel']= X_traincat_new['fuel'].astype("object")
X_traincat_new["seller_type"] = X_traincat_new["seller_type"].astype("object")
X_traincat_new["transmission"] = X_traincat_new["transmission"].astype("object")
X_traincat_new["owner"] = X_traincat_new["owner"].astype("object")
X_traincat_new["seats"]= X_traincat_new["seats"].astype("object")
#print(X_traincat_new.head(10))

transformer=make_column_transformer((OneHotEncoder(drop = "first", handle_unknown = "ignore"), ["fuel", "seller_type", "transmission", "owner", "seats"]), remainder = "passthrough")
transformed_nc=transformer.fit_transform(X_traincat_new)
transformed_df_nc=pd.DataFrame(transformed_nc, columns =transformer.get_feature_names_out())

model3_l2 = Ridge(alpha=7.01)
model3_l2.fit(transformed_df_nc, y_train)

#print("r2_score for y_train:", r2_score(y_train, model3_l2.predict(transformed_df_nc)))
#print("MSE for y_train:", MSE(y_train, model3_l2.predict(transformed_df_nc)))
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float



class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_test = pd.DataFrame([item]).map(lambda x: x[1])
    df_test.columns=["name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner",
                           "mileage", "engine", "max_power", "torque", "seats"]
    weird_features = ["mileage", "engine", "max_power"]
    for name in weird_features:
        df_test[name] = df_test[name].str.split().str[0]
    df_test.drop(labels="torque", axis=1, inplace=True)
    for name in weird_features:
        df_test[name] = pd.to_numeric(df_train[name], errors="coerce")
    columns = ["mileage", "engine", "max_power", "seats"]
    for col in columns:
        df_test[col] = df_test[col].fillna(df_train[col].median())
    df_test["engine"] = df_test["engine"].astype(int)
    df_test["seats"] = df_test["seats"].astype(int)
    X_test_cat = df_test.drop(["selling_price", "name"], axis=1, inplace=False)
    X_testcat_new = X_test_cat.copy()
    X_testcat_new["square of year"] = X_testcat_new["year"] * X_testcat_new["year"]
    X_testcat_new = X_testcat_new.drop(labels="year", axis=1)
    name_obj = ["fuel", "seller_type", "transmission", "owner", "seats"]
    for name in name_obj:
        X_traincat_new[name] = X_traincat_new[name].astype("object")
    transformed_nc_test = transformer.transform(X_testcat_new)
    transformed_df_nc_test = pd.DataFrame(transformed_nc_test, columns=transformer.get_feature_names_out())
    return model3_l2.predict(transformed_df_nc_test)[0]

@app.post("/predict_items")
def predict_items(items: UploadFile) -> List[float]:
    content = items.file.read()
    buffer = BytesIO(content)
    df_test = pd.read_csv(buffer)
    df_test.columns=["name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner",
                           "mileage", "engine", "max_power", "torque", "seats"]
    weird_features = ["mileage", "engine", "max_power"]
    for name in weird_features:
        df_test[name] = df_test[name].str.split().str[0]
    df_test.drop(labels="torque", axis=1, inplace=True)
    for name in weird_features:
        df_test[name] = pd.to_numeric(df_train[name], errors="coerce")
    columns = ["mileage", "engine", "max_power", "seats"]
    for col in columns:
        df_test[col] = df_test[col].fillna(df_train[col].median())
    df_test["engine"] = df_test["engine"].astype(int)
    df_test["seats"] = df_test["seats"].astype(int)
    X_test_cat = df_test.drop(labels=["selling_price", "name"], axis=1, inplace=False)
    X_testcat_new = X_test_cat.copy()
    X_testcat_new["square of year"] = X_testcat_new["year"] * X_testcat_new["year"]
    X_testcat_new = X_testcat_new.drop(labels="year", axis=1)
    name_obj = ["fuel", "seller_type", "transmission", "owner", "seats"]
    for name in name_obj:
        X_testcat_new[name] = X_testcat_new[name].astype("object")
    transformed_nc_test = transformer.transform(X_testcat_new)
    transformed_df_nc_test = pd.DataFrame(transformed_nc_test, columns=transformer.get_feature_names_out())
    return model3_l2.predict(transformed_df_nc_test)