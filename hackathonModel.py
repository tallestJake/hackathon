from transformers import pipeline
import pandas as pd
classifier = pipeline("zero-shot-classification")
sequence = pd.read_csv(r'~/Downloads/hackathon.csv')
candidate_labels = ["food"]
sum = 0
for i, row in sequence.iterrows():
    score = classifier(row['Descriptions'], candidate_labels)
    percent = score['scores'][0]
    check1 = percent >0.5 and row['Category'] == 'Food'
    check2 = percent <=0.5 and row['Category'] == 'Not Food'
    if check1 or check2:
        sum+=1
        print(' Text {} has a {}% probabilty of relating to food!'.format(row['Title'],round(percent*100,2)))
    else:
        print('Line {} Not accurate'.format(i+1))

print(sum/len(sequence))



