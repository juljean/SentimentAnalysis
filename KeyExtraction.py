import spacy
import pandas as pd


filename = 'CSV\\Bloomberg_tweets.csv'
filenameoutput = 'CSV\\BloombergUPD.csv'
wrongKey = 'Bloomberg'

def recordingKeys(filename, filenameoutput, wrongKey):
    data_list = []
    data = pd.read_csv(filename)
    for index, row in data.iterrows():
        data_list.append(row['Comments'])

    nlp = spacy.load("en_core_web_sm")
    final_list = []
    final = []

    #forming the Huge KeyWords list
    for i in range(len(data_list)):
        text = data_list[i]
        doc = nlp(text)
        final.extend(list(doc.ents))
        for word in range(len(final)):
            if str(final[word]) not in final_list and str(final[word])!=wrongKey and str(final[word]).isalpha():
                final_list.append(str(final[word]))
    print(final_list)
    data_keys = []
    for i in range(len(data_list)):

        append_text = " "
        text = data_list[i]
        doc = list(nlp(text))

        #Forming kewWords for each row
        for el in doc:
            if str(el) in final_list and str(el) not in append_text:
                append_text += (str(el) + " ")
        data_keys.append(append_text)

    data['KeyWord'] = data_keys

    data.to_csv(filenameoutput, index=False)

recordingKeys(filename, filenameoutput, wrongKey)