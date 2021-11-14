import pandas as pd
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import swifter
pd.options.display.max_colwidth = 5000
subreddit_list = ['canada', 'liberal', 'conservative','politics',\
"twoxchromosomes","showerthoughts","tifu"]
political = ['liberal', 'conservative','politics']
nonpolitical = ["twoxchromosomes","showerthoughts","tifu"]
# Read the feather files
data = pd.DataFrame()
for sub in subreddit_list:
    rf = pd.read_feather('../data/feather_files/RS_2019_'+sub+'_df.feather')
    if sub in political:
        rf['class'] = 'political'
    elif sub in nonpolitical:
        rf['class'] = 'nonpolitical'
    elif sub == 'canada':
        rf['class'] = 'test'
    else:
        rf['class'] = 'Unkown'
    data = pd.concat((data,rf))

# Removing posts with less than 10 characters in the body.
data = data[(data.selftext.astype(str).str.len()>10)].reset_index()
print(data.groupby('class').describe(percentiles=[.5]))

# combining titles with the post body.
data['text'] = data.title + " " + data.selftext
# deleting the previous two columns.
data.drop(columns=["title","selftext"],inplace=True)

def clean_text(text):
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
#--------------------------------------------------
data.text = data.text.swifter.allow_dask_on_strings(enable=True).apply(clean_text)
data.reset_index().drop(columns=['index'])\
    .to_feather('../data/feather_files/data2019clean.feather')
