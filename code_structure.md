The first part of the implementation is the scrapping, extraction, and cleaning of Reddit data. We split the implementation into 4 files. Below, we show how to run and alter each file. Some files take parameters, but for others, we need to change the value of the variables. 

We need to create the following folders so that the program work without errors. Otherwise, we need to change the paths in the code files.

 - /data/sumbissions_zst
 - /data/processed
 - /data/feather_files
 - /data/log
 - /data/images

For the examples below, we wrote ‘#’ before shell commands.

1. Download.py

The following file scrapes all Reddit submissions for the provided years. The file saves the data to /data/submissions_zst. There is one file per month (ie, 12 files for year provided year)

Example:  # python 1_download.py 2018 2019 2020

2. process_zst.py

This file process all the files found in /data/submissions_zst and extracts the submissions for the list of subreddits. 

The list of subreddit could be provided as a parameter or added as a variable. If no parameters are provided, the default list of subreddits is the one defined in the file.

The output files are saved as JSON objects under /data/processed. The script produces one file per each subreddit and year pair.

Example: 
```bash
# python 2_process_zst.py canada politics news
```
3. process_json.py

The following takes as input a name of a subreddit and a year. It processes the corresponding JSON file in /data/processed and transforms it to data-frame. The output is saved in /data/feather_files.

Example: 
```
# python 3_process_json.py canada 2018
```
4. clean_reddit.py

This file applies the cleaning steps mentioned in the report. The input is a year. The output is saved in /data/feather_files. There are three lists of subreddits in this file: political, nonpolitical, and test. If you want to change the default lists, you need to change the variables in the beginning of the file.

Example: 
```
# python 4_clean_reddit.py 2020
```
Extra: clean_all.py 

Since the model applies to any data, we developed a data cleaning program that applies to any data. To use it, we need to import the function and call it.

Example:

 ```python 
 from 4_clean_all import clean

….

cleaned_data = clean(data, x_label, y_label, output_filename)
```
Where data is a data-frame, x_label and y_label are the column names of the text data and category columns, and output_filename is the name of the output data. The data is saved to /data/feather_files. 
  
----------------------------
  
The second part is the model. The model is defined as a class in wp_model.py. There are four methods that can be called. 

1. fit(data, valid_size=.2) # The main function. It fits the training data and calculates the probabilities 

2. (optional) fit_CV(data, valid_size=.2, std=True, params=None) # applies cross-validation for different values of k and alpha. If std=True, params will be set to a default list.

3. train_valid_predict() # calculate the prediction scores for the training and validation sets.

4. predict(x) # returns the predictions of the provided data

5. test_predict(x) # similar to predict() with the addition of wordclouds and sample output.

The file 5_applying_wp_model.py contains an example of applying the model to the Reddit data.
Below, is an example of applying the model to any data.
```python
import pandas as pd
from wp_model import Proto # ensure that the file is in the same folder
from 6_clean_all import clean # the data cleaning function. 
data = pd.read_csv('...') # ensure that the data is in data frame format
cleaned_data = clean(data, 'x', 'y', 'output_file.feather') 
# The second and third inputs are the names of the predictor and response columns
# The last parameter is the name of the file where the clean data will be saved to.
model = Proto('label', k=1000, log_to_file=True, harmonic_pscore=True)
# This initializes a model object. The first parameter is a label. It can be any word, it's used to identify model output files and logs.
model.fit(cleaned_data)
# calculates the probabilities
model.train_valid_predict()
# predicts the labels for the training and validation data and produces the tables and graphs in this report.
preds = model.test_predict(test.X)
# returns predictions for the input text series.
model.fit_cv(data) # runs the cross-validation for the k value.
```
Moreover, 6_applying_Random_Forest.py applies cross-validation on the Reddit data and random forest machine learning model.

  
-----------
