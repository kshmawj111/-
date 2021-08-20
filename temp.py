import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import xgboost as xgb

from lightgbm import LGBMRegressor
import lightgbm as lgb

from catboost import Pool, CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from functools import reduce
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import seaborn as sns

DATA_PREFIX = './data/'
label = '등록차량수'
pd.options.display.float_format = '{:.5f}'.format

train_df = pd.read_csv(DATA_PREFIX + 'train.csv')
train_df = train_df[train_df.duplicated() == False]

test_df = pd.read_csv(DATA_PREFIX + 'test.csv')
test_df = test_df[test_df.duplicated() == False]

merged = pd.concat([train_df, test_df]).reset_index(drop=True)
merged = merged.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)': 'metro', '도보 10분거리 내 버스정류장 수': 'bus'}, axis=1)

merged['전용면적'] = merged['전용면적'].apply(lambda x: round(x, 0))
merged['전용면적'] = merged['전용면적'].apply(lambda x: x - x % 10 if x % 10 < 5 else x - ((x % 10) - 5))
merged['전용면적'] = merged['전용면적'].astype(np.int)

merged['임대료'] = merged['임대료'].replace("-", np.nan)
merged['임대보증금'] = merged['임대보증금'].replace("-", np.nan)

merged['임대보증금'] = merged['임대보증금'].astype(float)
merged['임대료'] = merged['임대료'].astype(float)

apt = merged[merged['임대건물구분'] == '아파트'].copy()
commercial = merged[merged['임대건물구분'] == '상가'].copy()

apt['metro'] = apt['metro'].fillna(0)

apt['price'] = (2 * apt['임대보증금'] * apt['임대료']) / (apt['임대보증금'] + apt['임대료']) # 조화평균
regional_median_df = apt.groupby(['지역', '전용면적']).median()
regions = regional_median_df.index.get_level_values(0).unique()
missing = apt[pd.isna(apt['price']) == True].copy()

for row in missing.iterrows():
    try:
        idx, row = row[0], row[1]

        val = regional_median_df.loc[(row['지역'], round(row['전용면적'], -1)), 'price'].copy()
        missing.loc[idx, 'price'] = val

        val = regional_median_df.loc[(row['지역'], round(row['전용면적'], -1)), '임대료'].copy()
        missing.loc[idx, '임대료'] = val

        val = regional_median_df.loc[(row['지역'], round(row['전용면적'], -1)), '임대보증금'].copy()
        missing.loc[idx, '임대보증금'] = val

    except:
        continue

apt.update(missing)
missing = missing[(pd.isna(missing['price'])) | (pd.isna(missing['임대료'])) | (pd.isna(missing['임대보증금']))].copy()
missing.head(3)

from sklearn.linear_model import Lasso  # L1 regularazation regression model
from copy import deepcopy

models = {}

for col in ['임대보증금', '임대료', 'price']:

    regression_models = {}

    for region in regions:
        temp = regional_median_df.dropna().loc[region, col].reset_index().to_numpy()[:-1].copy()
        model = Lasso(alpha=2, random_state=3)

        model.fit(temp[:, 0].reshape(-1, 1), temp[:, 1])

        regression_models[region] = deepcopy(model)

        del model

    models[col] = regression_models

for col in ['임대보증금', '임대료', 'price']:

    regions = apt['지역'].unique()
    regression_models = models[col]

    for r in regions:
        model = regression_models[r]
        temp = apt[apt['지역'] == r].copy()

        x = temp['전용면적'].values.reshape(-1, 1)
        preds = model.predict(x)
        temp.loc[:, col] = preds.copy()
        apt.update(temp)

apt['수도권'] = apt['지역'].apply(lambda x: 1 if x in ['서울특별시', '경기도'] else 0)
req_type = apt['자격유형'].value_counts().index[0]
apt['자격유형'] = apt['자격유형'].fillna(req_type)
apt['bus'] = apt['bus'].fillna(apt['bus'].median())

commercial_codes = commercial['단지코드']
apt['has_commercial'] = apt['단지코드'].apply(lambda x: 1 if x in commercial_codes.values else 0)

refined =  apt[apt['임대건물구분'] == '아파트'] # 상가 데이터는 전부 이상치로 행동할 수 있어서 삭제.

refined['bus'] = refined['bus'].apply(lambda x: 6 if x > 6 else x)
refined['공가수'] = refined['공가수'].apply(lambda x: 21 if x > 21 else x)
refined['metro'] = refined['metro'].apply(lambda x: 1 if x > 0 else x) # cat_feature

cat_cols = ['지역', '공급유형', '자격유형', 'metro', 'has_commercial']

# 지역을 제외한 각 피쳐의 상위 7개의 항목에 대해서만 보존, 나머지는 other로 매핑
counts = refined['공급유형'].value_counts()
supply_cat = counts.index[:7]
refined['공급유형'] = refined['공급유형'].apply(lambda x: x if x in supply_cat else 'other')

# 지역을 제외한 각 피쳐의 상위 7개의 항목에 대해서만 보존, 나머지는 other로 매핑
counts = refined['자격유형'].value_counts()
supply_cat = counts.index[:7]
refined['자격유형'] = refined['자격유형'].apply(lambda x: x if x in supply_cat else 'other')

refined = refined.drop('임대건물구분', axis=1) # 오직 아파트만 있으므로 없애도 됨

outlier_dropped = refined[(pd.isna(refined['등록차량수']) == True) | (refined['등록차량수'] <= 2200)].reset_index(drop=True)
price_dropped = refined.drop('price', axis=1).reset_index(drop=True)

commercial_codes = commercial['단지코드']
cat_features = ['지역', '공급유형', '자격유형', 'has_commercial', 'metro']

refined[cat_features] = refined[cat_features].astype(str)

from sklearn.preprocessing import LabelEncoder


def encode(df, cat_features, method='one_hot'):
    df = df.copy()

    if method == 'one_hot':
        if cat_features:
            df = pd.get_dummies(df, columns=cat_features)

    elif method == 'label':
        le = LabelEncoder()

        for cat in cat_features:
            df[cat] = le.fit_transform(df[cat])

    df = df.set_index("단지코드")
    print(df.shape)

    return df


from sklearn.model_selection import train_test_split


def split_dataset(df, random_state=6):
    train_idx = set(train_df['단지코드'].unique()) & set(df.index)
    test_idx = set(test_df['단지코드'].unique()) & set(df.index)

    train = df.loc[train_idx, :]
    test = df.loc[test_idx, :]
    test = test.drop('등록차량수', axis=1)

    y = train.pop('등록차량수')
    X = train.copy()

    print(X.shape, end=', ')
    print(y.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.07, random_state=random_state)
    return X_train, X_valid, y_train, y_valid, test

encoded_refined = encode(refined, cat_features, method=None)


refined_ds = split_dataset(encoded_refined)
X_train, X_valid, y_train, y_valid, test = refined_ds

train_pool = Pool(X_train,
                  y_train,
                  cat_features=cat_features)
test_pool = Pool(X_valid,
                 cat_features=cat_features)

# specify the training parameters
model = CatBoostRegressor(iterations=200,
                          max_depth=5,
                          verbose=50,
                          learning_rate=0.3,
                          loss_function='RMSE')

model.fit(train_pool)
# make the prediction using the resulting model
preds = model.predict(test_pool)
mae = np.mean(np.abs(y_valid - preds))
print(mae)



