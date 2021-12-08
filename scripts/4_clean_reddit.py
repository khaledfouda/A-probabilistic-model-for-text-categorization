
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import swifter
pd.options.display.max_colwidth = 5000

political = ['liberal', 'conservative', 'politics']
nonpolitical = ["twoxchromosomes", "showerthoughts", 'todayilearned', "tifu"]
test = ['canada']

subreddit_list = political + nonpolitical + test


# Read the feather files
def clean(year):
    data = pd.DataFrame()
    for sub in subreddit_list:
        rf = pd.read_feather('../data/feather_files/RS_' + year +
                             '_' + sub + '_df.feather')
        if sub in political:
            rf['tclass'] = 'political'
        elif sub in nonpolitical:
            rf['tclass'] = 'nonpolitical'
        elif sub == 'canada':
            rf['tclass'] = 'test'
        else:
            rf['tclass'] = 'Unkown'
        data = pd.concat((data, rf))

    # combining titles with the post body.
    data['text'] = data.title + " " + data.selftext
    # deleting the previous two columns.
    data.drop(columns=["title", "selftext"], inplace=True)
    # change letters to lowercase.
    data.text = data.text.str.lower()
    # remove the entry of deleted posts.
    data.text = data.text.replace({'[deleted]': '', '[removed]': '', 'http': '', 'tifu': '', 'todayilearned': ''})

    # Removing posts with less than 20 characters in the body.
    data = data[(data.text.astype(str).str.len() > 20)].reset_index(drop=True)
    print(data.groupby('tclass').describe(percentiles=[.5]))
    print(data.groupby('subreddit').describe(percentiles=[.5]))

    # functions for reducing words. Either stimming or lemmatizing. The latter is better.
    lemma = WordNetLemmatizer()
    english_stopwords = stopwords.words("english")

    def clean_word(word):
        """
        Input : text
        output : cleaned text
        process:
            1.Remove non-alphabitical words
            2.remove words of less than 3 characters
            3.Remove stopwords
            4.Transform words to lower characters
            4.lemmatize the text - First verbs then nouns
            * Steps are performed in that order.
        """
        word = lemma.lemmatize(word, pos='v')
        word = lemma.lemmatize(word, pos='n')
        word = lemma.lemmatize(word, pos='a')

        if (not word.isalpha()) or (len(word) <= 3) or (word in english_stopwords):
            return ''
        return word

    # --------------------------------------------------
    # applying clean_word() to every word.
    data.text = data.text.swifter.allow_dask_on_strings(enable=True).apply(
        lambda text: ' '.join(map(clean_word, text.split())))
    # ---------------------------------------------------
    # Remove data rows with empty text fields.
    indic = data.text.swifter.allow_dask_on_strings(enable=True).apply(lambda d: 0 if len(d.split()) == 0 else 1)
    indic = indic[indic == 0].index
    data = data.drop(index=indic)
    # -----------------------------------------
    # save to disk
    data.reset_index(drop=True) \
        .to_feather('../data/feather_files/data' + year + 'clean.feather')
    return data


if __name__ == '__main__':
    _ = clean('2019')
    _ = clean('2020')
