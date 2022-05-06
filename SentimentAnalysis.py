from nlpia.data.loaders import get_data
import pandas as pd
from KeyExtraction import Tokenization, stop_words, data_list, df
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

pd.set_option('display.width', 75)

nytdata = get_data('hutto_nyt')
bags_of_words = []

for text in nytdata.text:
    token = Tokenization(text)
    token_without_stopwords = [x for x in token if
                                x not in stop_words and x.isalpha()
                                and len(x) > 1 and x.islower()]
    bags_of_words.append(Counter(token_without_stopwords))

df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(float)

nb = MultinomialNB()
nb.fit(df_bows, nytdata.sentiment > 0)
nytdata['predicted_sentiment'] = nb.predict_proba(df_bows)[:, 1]* 4 - 2
nytdata['error'] = (nytdata.predicted_sentiment - nytdata.sentiment).abs()
nytdata.error.mean().round(1)
nytdata['sentiment_ispositive'] = (nytdata.sentiment > 0).astype(float)
nytdata['predicted_ispositive'] = (nytdata.predicted_sentiment > 0).astype(float)
print(nytdata['''sentiment predicted_sentiment sentiment_ispositive predicted_ispositive'''.split()].head(40))
print((nytdata.predicted_ispositive == nytdata.sentiment_ispositive).sum() / len(nytdata))

lem_tokens = []

for i in range(len(data_list)):
    token = (Tokenization(data_list[i]))
    token_without_stopwords = [x for x in token if
                                x not in stop_words and x.isalpha()
                                and len(x) > 1 and x.islower()]
    lem_tokens.append(Counter(token_without_stopwords))

df_final_bows = pd.DataFrame.from_records(lem_tokens)
df_all_bows = df_bows.append(df_final_bows)
df_final_bows = df_all_bows.iloc[len(nytdata):][df_bows.columns]
df_final_bows = df_final_bows.fillna(0).astype(float)

df["PredSent"] = nb.predict_proba(df_final_bows)[:, 1]* 2 - 1

print(df)