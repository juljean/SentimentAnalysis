import spacy
import pandas as pd
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from itertools import dropwhile
from textblob import TextBlob
from collections import Counter

# nltk.download('wordnet')
# nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')

filename = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\CSV\\BloombergUPD.csv'
filenameoutput = 'CSV\\BloombergUPD.csv'
punctuations = "?:!,.;()"
df = pd.read_csv(filename, usecols=['Comments'])


#Extracting the words from teh sentance & returning list

#Tokenization draft
# pattern = re.compile(r"([-\s.,;!?])+")

def Decapitalization(text):
    str = []
    for i in text.split(". "):
        if len(i)>1:
            str.append(i[0].lower() + i[1:])
    return '. '.join(str)

#print(Decapitalization("U.S. Mortgage Rates Hover at the Highest Level. Since 2010 U.S. mortgage rates were little changed, holding at the highest level since April 2010."))
def Purification():
    data_list = []
    element = "bloomberg.com"
    data = df['Comments'].tolist()
    for i in range (len(data)):
        if element in data[i]:
            data_list.append(" ".join(list(dropwhile(lambda x: x != element, data[i].split()))[1:]))
        else: data_list.append(data[i])
    return data_list

def KeysDef(tokens, filenameoutput):
    nlp = spacy.load("en_core_web_sm")
    final_list = []
    final = []

    #forming the Huge KeyWords list
    for i in range(len(tokens)):
        text = tokens[i]
        doc = nlp(text)
        final.extend(list(doc.ents))
    append_text = ""
    data_keys = []
    #Forming kewWords for each row
    for el in doc:
        if str(el) in final_list and str(el) not in append_text:
            append_text += (str(el) + " ")
    data_keys.append(append_text)

    df['KeyWord'] = data_keys

    df.to_csv(filenameoutput, index=False)


def Tokenization(data_text):
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    case_folded = Decapitalization(data_text)
    token = tokenizer.tokenize(case_folded)

    lem_token = [lemmatizer.lemmatize(str(i), pos="v") for i in token]
    lem_token = [lemmatizer.lemmatize(str(i), pos="a") for i in lem_token]
    lem_token = [lemmatizer.lemmatize(str(i), pos="n") for i in lem_token]
    return lem_token

def SentimentDef(lem_tokens):
    data_sent = []

    for i in range (len(lem_tokens)):
        tokens_without_stopwords = [x for x in lem_tokens[i] if
                                    x not in stop_words and x.isalpha()
                                    and len(x) > 1 and x.islower()]

        text = " ".join(tokens_without_stopwords)
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        data_sent.append(sentiment)

    df['Sentiment'] = data_sent

    print(df)
    #df.to_csv(filenameoutput, index=False)

#final_tokens = SentimentDef(Tokenization(Purification(filename)))

data_list = Purification()
