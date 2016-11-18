import nltk
from nltk.corpus import brown

data=brown.tagged_sents(categories=["adventure"],tagset="universal")

#This example to see how NaiveBayes with only one feature works. (The only feature is word by itself)
from NaiveBayes import NaiveBayesModel
model=NaiveBayesModel()
model.train(data)

first_sent=data[0]
first_sent_words=[w for w,l in first_sent]
prediction=model.predict(first_sent_words)
print(prediction)



# This example to see how Naive Bayes with different features works
# So it will have a feature extractor
from NaiveBayesUpdate import NaiveBayesModel_v2
import collections

model=NaiveBayesModel_v2()

def feature_extractor(word):
    feature_set={}
    feature_set["word"]=word
    return feature_set

formated_data=[]
for sent in data:
    tmp=[(feature_extractor(w),l) for w,l in sent]
    formated_data.extend(tmp)


model.train(formated_data)

first_sent_feature_set=[feature_extractor(w) for w,l in data[0]]
prediction=model.predict(first_sent_feature_set)
print(prediction)