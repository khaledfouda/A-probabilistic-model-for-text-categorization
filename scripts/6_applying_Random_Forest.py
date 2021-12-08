
if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    import pandas as pd
    import numpy as np

    pd.options.display.max_colwidth = 5000
    import logging.handlers
    import pickle
    from tabulate import tabulate

    # number of prototypical words to keep.
    k = 200
    log_to_file = True

    try:
        log.info("*******************************")
        log.info("Log is already initiated.")
    except:
        if log_to_file:
            logging.basicConfig(filename=f"../data/log/log_CV training__.log",
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
            log = logging.getLogger("Bot")
            log.info("###########################################################")

        else:
            log = logging.getLogger("Bot")
            log.setLevel(logging.DEBUG)
            log.addHandler(logging.StreamHandler())



    def SCORER(y_true, y_pred):
        scr = pd.Series(dtype=np.float32)
        scr["Accuracy"] = accuracy_score(y_true, y_pred)
        scr["f1_score"] = f1_score(y_true, y_pred)
        scr['Precision'] = precision_score(y_true, y_pred)
        scr['Recall'] = recall_score(y_true, y_pred)
        return scr


    # Read data file (2019&2020)
    data = pd.read_feather('../data/feather_files/pretraining/data2019clean.feather')
    data = pd.concat((data, \
                      pd.read_feather('../data/feather_files/pretraining/data2020clean.feather')))
    # Take test data out
    test = data.query("tclass == 'test'").reset_index(drop=True).rename(columns={'text': 'X'})[['X']]
    data = data.query("tclass == 'political' or tclass=='nonpolitical'").reset_index(drop=True)
    # data = data.sample(100000)
    data['X'] = data.text
    data['Y'] = pd.get_dummies(data.tclass).drop('nonpolitical', axis=1).values.ravel()
    data = data[['X', 'Y']]
    log.info("A data sample")
    log.info(data.head(1))
    log.info(f"The shape of the data is {data.shape}")
    log.info('Proportions of positive (political) and negative labels:')
    log.info(data.Y.value_counts() / data.shape[0])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data.X)
    Y = data.Y

    test_size = .2
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=test_size, random_state=100, shuffle=True,
                                                          stratify=data.Y)
    log.info(
        f"Taking {round(.2 * 100, 2)}% test subset. The resulting train shape is {X_train.shape} and test shape is {Y_valid.shape}")

    # best parameter
    best_fit = RandomForestClassifier(bootstrap=False, class_weight='balanced_subsample',
                                      max_depth=80, min_samples_leaf=2, min_samples_split=5, n_estimators=1000,
                                      n_jobs=4, verbose=10)

    best_fit.fit(X_train, Y_train)
    preds_v = best_fit.predict(X_valid)
    preds_t = best_fit.predict(X_train)
    scores = pd.DataFrame()
    scores[f'Train'] = SCORER(Y_train, preds_t)
    scores[f'Valid'] = SCORER(Y_valid, preds_v)
    log.info(tabulate(scores, headers='keys', tablefmt='psql'))
    log.info(tabulate(scores, headers='keys', tablefmt='latex_raw'))
    print(tabulate(scores, headers='keys', tablefmt='psql'))
    model_outfile = "../data/models/randomforest_06Dec.pickle"
    log.info(f"Saving trained model to {model_outfile}")
    pickle.dump(best_fit, open(model_outfile, 'wb'))
    # ---------------------------------------------------------------------------
    # ----- Cross validation ---------------------
    class_weight = ['balanced', 'balanced_subsample']
    n_estimators = [50, 100, 150, 500, 1000]
    max_features = ['auto', 'sqrt']
    max_depth = [10, 50, 80, 100, 120]
    min_samples_split = [2, 5, 6, 7, 10]
    min_samples_leaf = [1, 2, 4, 6]
    bootstrap = [True, False]
    param_grid = {'class_weight': class_weight,
                  'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                   n_iter=10, cv=5, random_state=6, n_jobs=-1, verbose=600)

    rf_random.fit(X_train, Y_train)
    optimized_rf = rf_random.best_estimator_
