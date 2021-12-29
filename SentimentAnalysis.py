from textblob import TextBlob
import pandas as pd

filename = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\BloombergUPD.csv'

def findingSentiment(filename):
    data_list = []
    data = pd.read_csv(filename)
    for index, row in data.iterrows():
        data_list.append(row['Comments'])

    data_sent = []
    for i in range(len(data_list)):
        text = data_list[i]
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        data_sent.append(sentiment)

    data['Sentiment'] = data_sent

    print(data)
    data.to_csv(filename, index=False)

findingSentiment(filename)