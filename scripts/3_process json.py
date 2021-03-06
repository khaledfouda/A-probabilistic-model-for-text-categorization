"""
 This files turns the json files into one pandas feather file.
 One subreddit, One year, twelve files for each month (or less)
 A sample call would be:
 python 3_process_json.py  canada 2016
"""
# ------------------------------------
import os
import pandas as pd
import sys
import logging.handlers
# ---------------------------------------------
log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
# --------------------------------------------------------------


def parsefile(input_file):
    d = pd.read_json(input_file)
    # "Ups" is removed since not included in 2019 data.
    keep_cols = ['author', 'created_utc', 'title', 'num_comments',
                 'score', 'domain', 'selftext', 'locked', 'subreddit']
    d = d[keep_cols]
    d.created_utc = pd.to_datetime(d.created_utc, unit='s').dt.date
    d = d.loc[d.num_comments >= 15]
    return d
# ------------------------------------------------


if __name__ == '__main__':
    input_folder = '../data/processed/'
    subreddit = sys.argv[0].lower()
    year = sys.argv[1]
    output_folder = '../data/feather_files/'
    input_files = []
    total_size = 0
    output_file = ''
    # --------------------------------------------------
    for subdir, dirs, files in os.walk(input_folder):
        for filename in files:
            if str(year) not in filename:
                continue
            if str(subreddit) not in filename:
                continue
            if output_file == '':
                output_file = output_folder + '/' + str(filename.split('-')[0]) +\
                 "_" + str(subreddit) + '_df.feather'
            file_path = os.path.join(subdir, filename)
            input_files.append(file_path)
            total_size += os.stat(file_path).st_size
    # -----------------------------------------------------
    log.info(f"Processing {len(input_files)} files of {(total_size/(2**30)):.2f} gigabytes" +
             f" of the subreddit {subreddit} and year {year}\nsaving to {output_file}")
    # ----------------------------------------------------------
    data = pd.DataFrame()
    for file in input_files:
        d = parsefile(file)
        data = pd.concat((data, d))
    log.info("Done processing. Saving to disk ....")
    data.reset_index().drop(columns=['index']).to_feather(output_file)
    log.info("Data file saved successfully.")
    log.info("..DONE..")
    # ------------------------------------------------
