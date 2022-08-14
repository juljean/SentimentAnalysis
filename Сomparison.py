import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
word_vectors = Word2Vec.load("word2vec.model")

filename1 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\CSV\\BloombergUPD.csv'
filename2 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\CSV\\NYT_tweetsUPD.csv'
filename3 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\CSV\\WSJ_tweetsUPD.csv'
filename4 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\CSV\\WashingtonPostUPD.csv'

topic = 'Omicron'


def getData(filename, ans_polarity):
    data = pd.read_csv(filename)
    for index, row in data.iterrows():
        if topic in row['KeyWord'] : ans_polarity.append(row['Sentiment'])
    return ans_polarity

ans1 = (getData(filename1, []))
ans2 = (getData(filename2, []))
ans3 = (getData(filename3, []))
ans4 = (getData(filename4, []))

plt.plot(ans1, color='#2A363B', label='Bloomberg')
plt.plot(ans2, color='#E84A5F', label='NYT')
plt.plot(ans3, color='#99B898', label='WSJ')
plt.plot(ans4, color='#FF847C', label='WashingtonPost')


plt.title("SearchWord: "+ topic)
plt.legend()
plt.show()