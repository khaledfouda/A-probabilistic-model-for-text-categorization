

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from prototypical_fit_new import Proto

    # Read data file (2019&2020)
    data = pd.read_feather('../data/feather_files/pretraining/data2019clean.feather')
    data = pd.concat((data, pd.read_feather('../data/feather_files/pretraining/data2020clean.feather')))
    # Take test data out
    test = data.query("tclass == 'test'").reset_index(drop=True).rename(columns={'text': 'X'})[['X']]
    data = data.query("tclass == 'political' or tclass=='nonpolitical'").reset_index(drop=True)
    data['X'] = data.text
    data['Y'] = pd.get_dummies(data.tclass).drop('nonpolitical', axis=1).values.ravel()
    data = data[['X', 'Y']]
    print("A data sample")
    print(data.head(1))
    print(f"The shape of the data is {data.shape}")
    print('Proportions of positive (political) and negative labels:')
    print(data.Y.value_counts() / data.shape[0])

    ob = Proto('REDDIT_last', k=1000, harmonic_pscore=True, log_to_file=True)
    _ = ob.fit(data)
    # _ = ob.train_valid_predict()
    # _ = ob.test_predict(test.X)
