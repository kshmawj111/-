import copy

from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np


cat_data_sets = {}
cat_params = {}
cat_features = []

def cat_boost(ds, iterations,
              _cv=False,
              plot=True,
              return_test_pred=False,
              drop_cols=None,
              verbose=100,
              save=False,
              params=cat_params,
              add_cat_feat=[],
              n_folds=1,
              ):
    X_train, X_valid, y_train, y_valid, X_test = cat_data_sets[ds]
    _cat_features = cat_features + add_cat_feat

    if drop_cols:
        X_train = X_train.drop(drop_cols, axis=1)
        X_valid = X_valid.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)

    train_pool = Pool(X_train,
                      y_train,
                      cat_features=_cat_features)

    valid_pool = Pool(X_valid,
                      cat_features=_cat_features)

    test_pool = Pool(X_test,
                     cat_features=_cat_features)

    if _cv:
        p = 'True' if plot is True else 'False'
        scores = cv(train_pool,
                    nfold=10,
                    plot=p,
                    verbose=verbose,
                    params=cat_params,
                    iterations=iterations,
                    )

    else:
        model = CatBoostRegressor(iterations=iterations, verbose=verbose, cat_features=_cat_features, **cat_params)
        models = None

        if n_folds == 1:
            model.fit(train_pool, eval_set=[(X_valid, y_valid)])
            val_pred = model.predict(valid_pool)
            mae = np.round(np.mean(np.abs(y_valid.values - val_pred)), 4)

        else:
            _, models = kfold_run(X_train, X_test,
                                  n_splits=n_folds, iterations=iterations,
                                  verbose=verbose, cat_features=_cat_features,
                                  cat_params=cat_params)
            val_pred = kfold_predict(models, X_valid, y_valid)


        if plot:
            plot_df = pd.DataFrame(val_pred, index=X_valid.index).reset_index()
            plot_df['Truth'] = y_valid.values
            plot_df = plot_df.groupby("단지코드").median()
            plot_df = plot_df.sort_values('Truth')

            diff = np.abs(plot_df['Truth'] - plot_df[0])
            diff = diff.sort_values(ascending=False)
            print("top 5 error codes:")
            print(diff[:5])

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.scatterplot(x=plot_df.index, y=plot_df[0], ax=ax, markers='x')
            sns.scatterplot(x=plot_df.index, y=plot_df['Truth'], ax=ax, markers='o')

            ax.legend(["pred", "truth"], loc=0, frameon=True)
            ax.set_title(f"Catboost's (num_boost: {iterations}, mae: {mae})")
            ax.set_xticks([])
            ax.set_ylabel("value")

            plt.show()

        if return_test_pred:
            if n_folds == 1:
                test_preds = model.predict(test_pool)

            else:
                test_preds = kfold_predict(models, X_test)

            preds = pd.DataFrame(test_preds, index=X_test.index).reset_index()
            _preds = preds.groupby("단지코드").median()
            sample_sub = pd.read_csv(DATA_PREFIX + "sample_submission.csv")
            sample_sub['num'] = sample_sub['code'].apply(lambda x: _preds.loc[x, 0])
            sample_sub = sample_sub.set_index('code')

            if save:
                sample_sub.to_csv(f"./submissions/cat_boost_{n_folds}_{ds}_{iterations}_{mae}.csv")

            return sample_sub


def kfold_predict(models, X, y=None):
    iterable = True

    try:
        iter(models)

    except TypeError as e:
        iterable = False

    if iterable:
        test_pred = np.zeros((X.shape[0], len(models)))

        for i, m in enumerate(models):
            test_pred[:, i] = m.predict(X)

        test_pred_mean = np.median(test_pred, axis=1).reshape(-1, 1)

        if y:
            mae = np.mean(np.abs(test_pred_mean - y))
            print(f"MAE: {round(mae, 4)}")

        return test_pred_mean

    else:
        pred = models.predict(X)

        if y:
            mae = np.mean(np.abs(pred - y))
            print(f"MAE: {round(mae, 4)}")

        return pred



def kfold_run(X_train, y_train,
              n_splits=10,
              random_state=4,
              iterations=1000,
              verbose=1000,
              cat_features=None,
                cat_params = None,

              ):
    kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    train_fold_pred = np.zeros((X_train.shape[0], 1))

    X_train,  y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    models = []

    for i, (t_i, v_i) in enumerate(kfold.split(X_train)):
        x_t, y_t = X_train.loc[t_i, :], y_train.loc[t_i]
        val = X_train.loc[v_i, :]

        model = CatBoostRegressor(iterations=iterations, verbose=verbose, cat_features=cat_features, **cat_params)

        model.fit(x_t, y_t)

        models.append(copy.deepcopy(model))
        train_fold_pred[v_i, :] = model.predict(val).reshape(-1, 1)

    return train_fold_pred, models

