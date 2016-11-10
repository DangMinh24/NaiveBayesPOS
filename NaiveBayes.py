
import numpy as np
from nltk.corpus import brown
from nltk import defaultdict


class NaiveBayesModel():
    def __init__(self):
        self.count_wl=defaultdict(lambda :0)
        self.count_l=defaultdict(lambda :0)
        self.K_l=set()
        self.prob_wl=defaultdict(lambda :0)
        self.prob_l=defaultdict(lambda :0)

    def train(self,corpus,alpha=0.1):
        for sent in corpus:
            for w,l in sent:
                self.count_l[l]+=1
                wl=w+"_"+l
                self.count_wl[wl]+=1
                self.K_l.add(l)
        self.K_l=list(self.K_l)
        self.alpha=alpha

    def predict(self,sent):
        def smoothing(X, K_l, c_wl, alpha=0.1):
            for k, v in c_wl.items():
                c_wl[k] += alpha

            for l in K_l:
                for x in X:
                    c_wl[x + "_" + l] += alpha

            return c_wl

        self.count_wl=smoothing(sent,self.K_l,c_wl=self.count_wl,alpha=self.alpha)

        def probability(c_wl,c_l,alpha=0.1):
            prob_wl=defaultdict(lambda :0)
            prob_l=defaultdict(lambda :0)

            for wl in c_wl.keys():
                l=wl.split("_")[1]
                prob_wl[wl]=float(c_wl[wl])/(c_l[l]+alpha*len(c_wl))

            for l in c_l.keys():
                prob_l[l]=float(c_l[l])/np.sum(list(c_l.values()))

            return prob_wl,prob_l

        self.prob_wl,self.prob_l=probability(self.count_wl,self.count_l,self.alpha)

        prediction=[]
        for w in sent:
            result_tmp=[]
            result_label=[]
            for l in self.K_l:
                wl=w+"_"+l
                score=self.prob_wl[wl]*self.prob_l[l]
                result_tmp.append(score)
                result_label.append(l)
            best_label=result_label[np.argmax(result_tmp)]
            prediction.append(best_label)
        return prediction

    def predict_many(self,sents):
        predictions=[]
        for sent in sents:
            prediction=self.predict(sent)
            predictions.append(prediction)
        return predictions


