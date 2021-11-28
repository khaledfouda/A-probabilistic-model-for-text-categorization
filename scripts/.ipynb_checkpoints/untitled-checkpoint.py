import sys
sys.path.append('../scripts')
import pandas as pd
from prototypical_fit import fit

# Read data file (2019&2020)
data = pd.read_feather('../data/feather_files/data2019clean.feather')
data = pd.concat((data,\
        pd.read_feather('../data/feather_files/data2020clean.feather')))
# Take test data out
test = data.query("tclass == 'test'").reset_index(drop=True).rename(columns={'text':'X'})[['X']]
data = data.query("tclass == 'political' or tclass=='nonpolitical'").reset_index(drop=True)
data['X'] = data.text
data['Y'] = pd.get_dummies(data.tclass).drop('nonpolitical', axis=1).values.ravel()
data = data[['X','Y']]
print("A data sample")
print(data.head(1))
print(f"The shape of the data is {data.shape}")
print('Proportions of positive (political) and negative labels:')
print(data.Y.value_counts() / data.shape[0]) 


fit(data, 'Reddit', k=900, log_to_file=True)