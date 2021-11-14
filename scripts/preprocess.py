import pandas as pd
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
pd.options.display.max_colwidth = 5000

lib19 = pd.read_feather('../data/feather_files/RS_2019_liberal_df.feather')
con19 = pd.read_feather("../data/feather_files/RS_2019_conservative_df.feather")
pol19 = pd.read_feather('../data/feather_files/RS_2019_politics_df.feather')
can19 = pd.read_feather('../data/feather_files/RS_2019_canada_df.feather')

pol19_sub = pol19[(pol19.selftext.astype(str).str.len()>10)].reset_index()
can19_sub = can19[(can19.selftext.astype(str).str.len()>10)].reset_index()
lib19_sub = lib19[(lib19.selftext.astype(str).str.len()>10)].reset_index()
con19_sub = con19[(con19.selftext.astype(str).str.len()>10)].reset_index()

def concat_text(df):
    df['text'] = df.title + " " + df.selftext
    df.drop(columns=["title","selftext"],inplace=True)
    return df
pol19_sub = concat_text(pol19_sub)
con19_sub = concat_text(con19_sub)
lib19_sub = concat_text(lib19_sub)
can19_sub = concat_text(can19_sub)

def clean_text(text):
    """
    To be filled later
    """
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    def clean(word):
        if not word.isalpha() or len(word) < 3:
            return False
        if word.lower() in stopwords.words("english"):
            return False
        return True
    tokens = " ".join(str(x) for x in \
        [lemma.lemmatize(\
            lemma.lemmatize(word.lower(),pos="v")\
            ,pos="n") for word in tokens if clean(word)]\
        )
    return tokens

can19_sub.text = can19_sub.text.map(clean_text)
pol19_sub.text = pol19_sub.text.map(clean_text)
con19_sub.text = con19_sub.text.map(clean_text)
lib19_sub.text = lib19_sub.text.map(clean_text)
