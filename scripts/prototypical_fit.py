import pandas as pd
import numpy as np
import nltk
import modin
import swifter
import collections
from importlib import reload
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
import logging.handlers
from prettytable import PrettyTable
from datetime import date
from sklearn.metrics import f1_score, accuracy_score
from tabulate import tabulate
pd.options.display.max_colwidth = 5000

# ---------------------------------------------
# Initiating logfile:


def init_log(k, label, log_to_file=False):
    reload(logging)
    log = logging.getLogger("Bot")
    if log.hasHandlers():
        log.handlers.clear()
    logging.shutdown()
    if log_to_file:
        logging.basicConfig(filename=f"../data/log/{label}_log_k={k}__{date.today().strftime('%d-%m-%Y')}__.log",
                            filemode='a',
                            format='%(asctime)s: %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        log = logging.getLogger("Bot")
        log.addHandler(logging.StreamHandler())
        log.info("######################################################################################")

    else:
        log = logging.getLogger("Bot")
        log.setLevel(logging.DEBUG)
        log.addHandler(logging.StreamHandler())
    return log
# --------------------------------------------------


def wordcloudGen(text, title, outfile, log):
    wordcloud = WordCloud(background_color="white", max_words=5000,
                          contour_width=6, contour_color='steelblue', scale=3)
    # Generate the first
    log.info("Word Cloud")
    log.info(title)
    wordcloud.generate(text)
    wordcloud.to_image()
    wordcloud.to_file(outfile)
    return
# ------------------------------------------------------


def GetCounters(train, valid, k, log, label):
    log.info("dividing data into classes")
    y1 = train.query("Y == 1").X
    y0 = train.query("Y == 0").X
    y1_valid = valid.query("Y==1").X
    y0_valid = valid.query("Y==0").X
    # Get a long string of all text for each of the categories to analyse
    log.info("Joining the series of text into one string per category")
    long_str_1 = ' '.join(y1)
    long_str_0 = ' '.join(y0)
    long_str_1_v = ' '.join(y1_valid)
    long_str_0_v = ' '.join(y0_valid)
    # Transform that string into a list of strings
    log.info("Dividing those long strings into lists of words")
    tokens_1 = long_str_1.split()
    tokens_0 = long_str_0.split()
    tokens_1_v = long_str_1_v.split()
    tokens_0_v = long_str_0_v.split()
    # --------------------------------------------------
    # Generate wordclouds
    wordcloudGen(long_str_1 + long_str_1_v, 'class 1', f'../data/images/{label}_class1_k={k}.png', log)
    wordcloudGen(long_str_0 + long_str_0_v, 'class 0', f'../data/images/{label}_class0_k={k}.png', log)
    # ----------------------------------------------
    # Counting the occurrences of each of the words. Output: list (word, #occurences)
    log.info("Counting the occurrences of each word per class")
    log.info(f"Total umber of words (training+validation) is:")
    log.info(f"Class 1: {len(tokens_1) + len(tokens_1_v)}, Class 0: {len(tokens_0) + len(tokens_0_v)}")
    counter_1 = collections.Counter(tokens_1)
    counter_0 = collections.Counter(tokens_0)
    log.info(f"Number of distinct training words for each class is")
    log.info(f"{len(counter_1)} and {len(counter_0)}")
    log.info("Visualizing the top 15 common words in each category [latex code below]")
    table = PrettyTable()
    table.title = 'Most common 15 words in each category '
    table.add_column('class 1', np.transpose(collections.Counter(tokens_1 + tokens_1_v).most_common(15))[0])
    table.add_column('class 0', np.transpose(collections.Counter(tokens_0 + tokens_0_v).most_common(15))[0])
    log.info(table)
    log.info(table.get_latex_string())
    return counter_1, counter_0
# -----------------------------------------------------------------------------


def WordDict(counter_1, counter_0, k, n_drop, log):
    log.info("Creating a table of the number of occurrences of each word in each of the two classes. Saved as worddict")
    # Make a table of word - #occurances in class 1 - #occurances in class 2 [saving to file]
    worddict = pd.DataFrame({'word': counter_1.keys(), 'c1_occ': counter_1.values()})
    worddict['c0_occ'] = worddict.word.swifter.apply(lambda x: counter_0.get(x, 0))
    # ----------------------------- class 2
    temp = pd.DataFrame({'word': counter_0.keys(), 'c0_occ': counter_0.values()})
    temp['c1_occ'] = temp.word.swifter.apply(lambda x: counter_1.get(x, 0))
    # Merge
    worddict = pd.concat((worddict, temp)).drop_duplicates('word').sort_values(by='c1_occ', ascending=False)
    # ------------------------------------------------------------------
    # log
    log.info(f"Number of distinct words is {len(worddict)}")
    log.info(f"Dropping words occurring less than {n_drop} times")
    # Drop words occurring less than n times
    worddict = worddict.query('c1_occ >= @n_drop or c0_occ >= @n_drop').reset_index(drop=True)
    log.info(f" number of words occurring at least {n_drop} times is {len(worddict)} words")

    # Computer score a
    log.info("Computing proto score. Equation 1. Objective, choose top k words")
    worddict['ssum'] = worddict.c1_occ + worddict.c0_occ
    worddict['sc_1'] = worddict.c1_occ / worddict.ssum
    worddict['sc_0'] = worddict.c0_occ / worddict.ssum
    # Remove non-words (ie. names)
    log.info("Removing names and other non-recognizable words.")
    log.info("Unfortunately, some names would be detected since they hold a second meaning")
    worddict = worddict[worddict.word.isin(nltk.corpus.words.words('en'))]
    # Keep top k words in each class -
    # k = 400
    log.info(f"keeping top {k} words in each class")
    temp = worddict.sort_values(by='sc_1', ascending=False)[0:k]
    temp['Y'] = 1
    worddict = worddict.sort_values(by='sc_0', ascending=False)[0:k]
    worddict['Y'] = 0
    worddict = pd.concat((temp, worddict))
    worddict = worddict.drop_duplicates('word') \
        .sort_values(by='sc_1', ascending=False).reset_index(drop=True)
    log.info(f"Number of words after keeping top {k} words is {len(worddict)}")
    # save to disk
    # worddict.to_feather(f'../data/feather_files/{label}_worddict_k={k}.feather')
    return worddict
# -----------------------------------------------------------


def Proto(train, worddict, sum_proto, k, log):
    # wp_in_u includes the frequency of the selected words in each of the posts
    # the shape is list of lists. posts x words
    log.info("Counting the occurrence of the chosen words inside each of the posts.")
    log.info("The resulting dataframe is of shape (number of posts)x(2k).")
    wp_in_u = train.X.swifter.apply(lambda d: ([d.count(wp) for wp in worddict.word]))
    wp_in_u = pd.DataFrame(wp_in_u)
    log.info(f"saved as wp_in_u_{k} with the dimension of {wp_in_u.shape}")
    log.info(f"Transforming the previous variable into a dataframe. saved as wp_proto_{k}")
    proto = wp_in_u.swifter.apply(lambda d: pd.Series(d[0]), axis=1)
    proto.columns = worddict.word
    proto.index = train.index
    # sum of all words in each post
    log.info("Creating the first set of features, equation 2.")
    log.info("(#occurence of wp in u)/(#words in u). A table of (#posts)x(2k)")
    proto = proto.divide(sum_proto, axis=0)
    # proto.to_feather(f'../data/feather_files/{label}_proto_{k}.feather')
    log.info(f"saved as proto_{k}")
    return proto
# ---------------------------------------------------------


def ProtoTrain(proto, worddict, sum_proto, train, k, log):
    log.info("The next feature is a score per (post,class)")
    proto_train = pd.DataFrame()
    # The numerator
    proto_train['sc_1'] = proto[worddict.query("Y == 1").word].sum(axis=1)
    proto_train['sc_0'] = proto[worddict.query("Y == 0").word].sum(axis=1)
    # Divide by the denominator (same as the previous score)
    proto_train = proto_train.divide(sum_proto, axis=0)
    proto_train['Y'] = train.Y
    proto_train['Y_pred'] = (proto_train.sc_1 > proto_train.sc_0).astype(int)
    # proto_train['accuracy'] = (proto_train.Y == proto_train.Y_pred).astype(int)
    proto_train['nonresp_flag'] = 0
    proto_train.loc[proto_train.eval('sc_1 == sc_0 == 0'), 'nonresp_flag'] = 1
    log.info('saving to disk')
    # save to disk
    # proto_train.to_feather(f'../data/feather_files/{label}_proto_train_{k}.feather')
    log.info(f"saved as proto_train_{k}")
    log.info(tabulate(proto_train.sample(3), headers='keys', tablefmt='psql'))
    return proto_train
# ---------------------------------------------------------


def ProtoCWP(proto, proto_train, k, log, label):
    log.info("Computing the probabilities of classes given proto words")
    log.info("The Y is an assignment to the class of higher probability")
    # probabilities of class given a proto word.
    # the Y is an  assignment to the class of higher probability
    # Rows are words
    proto_cwp = pd.DataFrame()

    proto_cwp['sc_1'] = proto.swifter.apply(lambda d: d.mul(proto_train.sc_1).sum())
    proto_cwp['sc_0'] = proto.swifter.apply(lambda d: d.mul(proto_train.sc_0).sum())
    proto_cwp['Y'] = proto_cwp.swifter.apply(lambda d: d.sc_1 > d.sc_0, axis=1)
    proto_cwp['word'] = proto.columns
    log.info("dataframe was created successfully. Saving to disk...")
    log.info(tabulate(proto_cwp.sample(3), headers='keys', tablefmt='psql'))
    proto_cwp.reset_index(drop=True).to_feather(f'../data/feather_files/{label}_proto_CWP_k={k}.feather')
    log.info('saved to disk')
    return proto_cwp
# ---------------------------------------------------------


def ProtoValid(valid, proto_cwp, k, log):
    log.info("Computing validation set predictions")
    proto_valid = pd.DataFrame()
    log.info("The class prob of a post is the sum of class|word probabilities for all proto word in class and in post")

    temp = proto_cwp[['sc_1', 'sc_0']]
    proto_valid[['sc_1', 'sc_0']] = valid.X.swifter.apply(
        lambda d: temp[proto_cwp.word.isin(d.split())].sum())[['sc_1', 'sc_0']]

    proto_valid['Y'] = valid.Y  # 1 for p and 0 for np

    proto_valid['Y_pred'] = (proto_valid.sc_1 > proto_valid.sc_0).astype(int)  # 1 for p and 0 for np
    # proto_valid['accuracy'] = (proto_valid.Y == proto_valid.Y_pred).astype(int)
    proto_valid['nonresp_flag'] = 0
    proto_valid.loc[proto_valid.eval('sc_1 == sc_0 == 0'), 'nonresp_flag'] = 1
    log.info('saving to disk')
    # save to disk
    # proto_valid.to_feather(f'../data/feather_files/{label}_proto_valid_{k}.feather')
    log.info(f"saved as proto_valid_{k}")
    log.info(tabulate(proto_valid.sample(3), headers='keys', tablefmt='psql'))
    return proto_valid
# ---------------------------------------------------------------------


def SCORER(y_true, y_pred, flag):
    scr = pd.Series(dtype=np.float32)
    scr["nonresponse"] = flag.value_counts().loc[1] / len(flag)
    scr["f1_score"] = f1_score(y_true, y_pred)
    scr["f1_score_a"] = f1_score(y_true[flag == 0], y_pred[flag == 0])
    scr["Accuracy"] = accuracy_score(y_true, y_pred)
    scr["Accuracy_a"] = accuracy_score(y_true[flag == 0], y_pred[flag == 0])
    return scr

# -------------------------------------------------------------


def fit(data, label, k, log_to_file=False, valid_size=.2, n_drop=50):
    log = init_log(k, label, log_to_file)
    log.info("Starting fit")
    log.info("A data sample")
    log.info(tabulate(data.sample(3), headers='keys', tablefmt='psql'))
    log.info(f"The shape of the data is {data.shape}")
    log.info(data.Y.value_counts() / data.shape[0])
    # ---------------------
    # Splitting to train and test
    train, valid = train_test_split(data, test_size=valid_size, random_state=100, shuffle=True, stratify=data.Y)
    # data=None
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    log.info(f"Taking {round(.2 * 100, 2)}% test subset.")
    log.info(f"The resulting train shape is {train.shape} and test shape is {valid.shape}")
    # ----------------------------------
    # transforming text
    # get text columns for training datasets
    counter_1, counter_0 = GetCounters(train, valid, k, log, label)
    # ----------------------------------------------------------------
    # Creating worddict
    worddict = WordDict(counter_1, counter_0, k, n_drop, log)
    # creating wordclouds of the new worddict
    wordcloudGen(' '.join(worddict.query("Y == 1").word), 'Top of class 1',
                 f'../data/images/{label}_top_class1_k={k}.png', log)
    wordcloudGen(' '.join(worddict.query("Y == 0").word), 'Top of class 0',
                 f'../data/images/{label}_top_class0_k={k}.png', log)
    # -------------------------------------------------------------------
    log.info("Computing the sum of words in each post")
    sum_proto = train.X.swifter.apply(lambda d: len(d.split()))
    # create proto
    proto = Proto(train, worddict, sum_proto, k, log)
    # create proto_train
    proto_train = ProtoTrain(proto, worddict, sum_proto, train, k, log)
    # create proto_cwp
    proto_cwp = ProtoCWP(proto, proto_train, k, log, label)
    # creating proto_valid
    proto_valid = ProtoValid(valid, proto_cwp, k, log)
    # ----------------------------------------------------------------------
    # Evaluating the model
    scores = pd.DataFrame()
    scores[f'Train_{k}'] = SCORER(proto_train.Y, proto_train.Y_pred, proto_train.nonresp_flag)
    scores[f'Valid_{k}'] = SCORER(proto_valid.Y, proto_valid.Y_pred, proto_valid.nonresp_flag)
    scores.reset_index().to_feather(f'../data/feather_files/{label}_scores_k={k}.feather')
    log.info(tabulate(scores, headers='keys', tablefmt='psql'))
    log.info(tabulate(scores, headers='keys', tablefmt='latex_raw'))
    # -------------------------------------------------------------------------------
    log.info("End")
# ---------------------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
