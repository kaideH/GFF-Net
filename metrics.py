import numpy as np
import random
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, cohen_kappa_score

class Metrics():

    def __init__(self, num_class):

        self.num_class = num_class
        """
                    | matrix matrix matrix
            predict | matrix matrix matrix
                    | matrix matrix matrix
                    +----------------------
                             labels
        """
        self._confusion_matrix = np.zeros((num_class, num_class)) # [predict, label]
        """
            softmax_scores / labels: for ROC / AUC
            [
                [score1, score2, score3],
                ...
                [score1, score2, score3],
            ]
        """
        self._labels = []
        self._softmax_scores = []
        self._predictions_list = []
        self._labels_list = []
        return

    def add_sample(self, predict, label, scores):
        self._confusion_matrix[predict, label] += 1

        self._softmax_scores.append(scores)
        one_hot = [0] * len(scores)
        one_hot[label] = 1
        self._labels.append(one_hot)

        self._predictions_list.append(predict)
        self._labels_list.append(label)
        return

    def index_check(self, index):
        assert index < self.num_class, f"error index {index}, the index must lower than {self.num_class}"
        return

    def confusion_matrix(self):
        return self._confusion_matrix

    def TP(self, index):
        self.index_check(index)
        return self._confusion_matrix[index, index]
    
    def FP(self, index):
        self.index_check(index)
        return np.sum(self._confusion_matrix[index, :]) - self.TP(index)
    
    def FN(self, index):
        self.index_check(index)
        return np.sum(self._confusion_matrix[:, index]) - self.TP(index)

    def TN(self, index):
        self.index_check(index)
        return np.sum(self._confusion_matrix) - self.TP(index) - self.FP(index) - self.FN(index)

    def acc(self, index=-1):
        if index == -1:
            return np.sum(np.diagonal(self._confusion_matrix)) / np.sum(self._confusion_matrix)
        else:
            return (self.TP(index) + self.TN(index)) / (self.TP(index) + self.TN(index) + self.FP(index) + self.FN(index))


    def sensitivity(self, index): # 敏感性(召回率), TP/(TP+FN)
        return self.TP(index) / (self.TP(index) + self.FN(index))

    def specificity(self, index): # 特异度, TN/(TN+FP)
        return self.TN(index) / (self.TN(index) + self.FP(index))

    def precision(self, index=-1): # 精确度, TP/(TP+FP)
        if index == -1:
            return precision_score(self._labels_list, self._predictions_list, average="weighted")
        else:
            return self.TP(index) / (self.TP(index) + self.FP(index))

    def recall(self, index=-1): # 召回率, TP/(TP+FN)
        if index == -1:
            return recall_score(self._labels_list, self._predictions_list, average="weighted")
        else:
            return self.sensitivity(index)

    def f1(self, index=-1): # F1 score, 2*Precision*recall/(Precision+recall)
        if index == -1:
            return f1_score(self._labels_list, self._predictions_list, average="weighted")
        else:
            return 2 * self.precision(index) * self.recall(index) / (self.precision(index) + self.recall(index))

    def kappa(self):
        return cohen_kappa_score(self._labels_list, self._predictions_list)

    def AUC_ROC(self, index):
        self.index_check(index)

        softmax_scores = np.array(self._softmax_scores)
        labels = np.array(self._labels)

        y_true = labels[:, index]
        y_score = softmax_scores[:, index]

        fpr, tpr, thread = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score


    

if __name__ == "__main__":
    metrics = Metrics(3)

    for i in range(100):
        a = random.random()
        b = random.uniform(0, 1 - a)
        c = 1 - a - b
        metrics.add_sample(random.randint(0, 2), random.randint(0, 2), [a, b, c])

    print(metrics.confusion_matrix())
    print(f"TP@0: {metrics.TP(0)}, FP@0: {metrics.FP(0)}, TN@0: {metrics.TN(0)}, FN@0: {metrics.FN(0)}")
    print(f"TP@1: {metrics.TP(1)}, FP@1: {metrics.FP(1)}, TN@1: {metrics.TN(1)}, FN@1: {metrics.FN(1)}")
    print(f"TP@2: {metrics.TP(2)}, FP@2: {metrics.FP(2)}, TN@2: {metrics.TN(2)}, FN@2: {metrics.FN(2)}")
    
    print(f"acc: {metrics.acc()}")
    print(f"acc@0: {metrics.acc(0)}, acc@1: {metrics.acc(1)}, acc@2: {metrics.acc(2)}")
    print(f"sensitivity@0: {metrics.sensitivity(0)}, sensitivity@1: {metrics.sensitivity(1)}, sensitivity@2: {metrics.sensitivity(2)}")
    print(f"specificity@0: {metrics.specificity(0)}, specificity@1: {metrics.specificity(1)}, specificity@2: {metrics.specificity(2)}")

    print(f"recall: {metrics.recall()}")
    print(f"recall@0: {metrics.recall(0)}, recall@1: {metrics.recall(1)}, recall@2: {metrics.recall(2)}")
    print(f"precision: {metrics.precision()}")
    print(f"precision@0: {metrics.precision(0)}, precision@1: {metrics.precision(1)}, precision@2: {metrics.precision(2)}")
    
    print(f"f1: {metrics.f1()}")
    print(f"f1@0: {metrics.f1(0)}, f1@1: {metrics.f1(1)}, f1@2: {metrics.f1(2)}")
    print(f"AUC@0: {metrics.AUC_ROC(0)[2]}, AUC@2: {metrics.AUC_ROC(2)[2]}")
    print(f"kappa: {metrics.kappa()}")

    # print(metrics.AUC_ROC(0)[0])







    