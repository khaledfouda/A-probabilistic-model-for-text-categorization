'''
 This files turns the json files into one pandas feather file.
 One subreddit, One year, twelve files for each month (or less)
 A sample call would be:
 combine_subset_todf.py ../data/processed/ canada 2016 ../data/feather_files/
'''
#------------------------------------
import os
import pandas as pd
import sys
import logging.handlers
#---------------------------------------------
log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
#--------------------------------------------------------------
def parsefile(input_file):
    d = pd.read_json(input_file)
    keeped_cols = ['author', 'created_utc', 'title', 'num_comments',
                  'score', 'ups', 'domain', 'selftext', 'locked', 'subreddit']
    d = d[keeped_cols]
    d.created_utc = pd.to_datetime(d.created_utc, unit='s').dt.date
    d = d.loc[d.num_comments>=15]
    return d
#------------------------------------------------
if __name__ == '__main__':
    input_folder = sys.argv[1]
    subreddit = sys.argv[2]
    year = sys.argv[3]
    output_folder = sys.argv[4]
    input_files = []
    total_size = 0
    output_file = ''
    #--------------------------------------------------
    for subdir, dirs, files in os.walk(input_folder):
        for filename in files:
            if str(year) not in filename:
                continue
            if str(subreddit) not in filename:
                continue
            if output_file == '':
                output_file = output_folder + '/' +str(filename.split('-')[0]) +\
                 str(subreddit) + 'df.feather'
            file_path = os.path.join(subdir, filename)
            input_files.append(file_path)
            total_size += os.stat(file_path).st_size
    #-----------------------------------------------------
    log.info(f"Processing {len(input_files)} files of {(total_size/(2**30)):.2f} gigabytes"+\
    f" of the subbreddit {subreddit} and year {year}\nsaving to {output_file}")
    #----------------------------------------------------------
    data = pd.DataFrame()
    for file in input_files:
        d = parsefile(file)
        data = pd.concat((data, d))
    log.info("Done processing. Saving to disk ....")
    data.reset_index().drop(columns=['index']).to_feather(output_file)
    log.info("Data file saved successfully.")
    log.info("..DONE..")
    #------------------------------------------------
