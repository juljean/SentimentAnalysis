import pandas as pd
import matplotlib.pyplot as plt

filename1 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\BloombergUPD.csv'
filename2 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\NYT_tweetsUPD.csv'
filename3 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\WSJ_tweetsUPD.csv'
filename4 = 'C:\\Users\\Jul\\PycharmProjects\\operationResearch\\WashingtonPostUPD.csv'

topic = 'Apple'


def getData(filename, ans_polarity):
    data = pd.read_csv(filename)
    for index, row in data.iterrows():
        if topic in row['KeyWord'] : ans_polarity.append(row['Sentiment'])
    return ans_polarity

ans1 = (getData(filename1, []))
ans2 = (getData(filename2, []))
ans3 = (getData(filename3, []))
ans4 = (getData(filename4, []))

plt.plot(ans1, color='r', label='Bloomberg')
plt.plot(ans2, color='g', label='NYT')
plt.plot(ans3, color='b', label='WSJ')
plt.plot(ans4, color='y', label='WashingtonPost')


plt.title("SearchWord: "+ topic)
plt.legend()
plt.show()