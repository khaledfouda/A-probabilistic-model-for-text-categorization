import pandas as pd
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer,pos_tag, PorterStemmer
from nltk.corpus import stopwords, wordnet
import swifter
from tabulate import tabulate
pd.options.display.max_colwidth = 5000

# ------------------------------------------------------------


def clean(data, x_label, y_label, outfilename):
    data['X'] = data[x_label]
    data['Y'] = data[y_label]
    tmp = data[['X', 'Y']]
    data = tmp
    del tmp
    # change letters to lowercase.
    data.X = data.X.str.lower()
    # convert y labels into binary. Expects only 2 labels
    data.Y = pd.get_dummies(data.Y).iloc[:, 0].values.ravel()
    # Removing posts with less than 20 characters in the body.
    data = data[(data.X.astype(str).str.len() > 20)].reset_index(drop=True)
    print(tabulate(data.groupby('Y').describe(percentiles=[.5]), headers='keys', tablefmt='psql'))

    # functions for reducing words. Either stimming or lemmatizing. The latter is better.
    lemma = WordNetLemmatizer()
    english_stopwords = stopwords.words("english")

    def cleanWord(word):
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
    data.X = data.X.swifter.allow_dask_on_strings(enable=True).apply(
        lambda d: ' '.join(map(cleanWord, d.split())))
    # ---------------------------------------------------
    # Remove data rows with empty text fields.
    indic = data.X.swifter.allow_dask_on_strings(enable=True).apply(lambda d: 0 if len(d.split()) == 0 else 1)
    indic = indic[indic == 0].index
    data = data.drop(index=indic)
    # -----------------------------------------
    # save to disk
    data.reset_index(drop=True)\
        .to_feather(f'../data/feather_files/{outfilename}.feather')
    return data
# -----------------------------------------------------------------------------------


if __name__ == '__main__':
    pass
