# Data Classification Algorithms 

Description: Implementation of classification algorithms in 2020 for the elective course "Pattern Recognition" for my undergraduate studies in Computer Science and 
Engineering at University of Ioannina. 

**Programming language: Python**

**Interpreter: Python 3.6**

**Software used: PyCharm (vers2019.3.5)**

------------------------------------------------------------------------------------------------------------------------------------

The two data experiments named "spambase.data" and "credit card clients dataset.xls" are problems of binary classification. 
The above experiments will be utilized for the needs of this assignment.

Classification methods that are being utilized:
1) LVQ algorithm with the specific variation of dynamically spliting the space od the data into sherical Gaussian areas of the same 
category. The process of the classificated decision for a foreign element will be determined by the category of the nearest 
Gaussian area.
2) Nearest Neighbor k-NN algorithm utilizing the Euclidean distance (variable k must be determined by the user each time)
3) Neural Network with Sigmoid as the activation function consisted of a) 1 hidden layer and different number (K) of neurons and b) 
2 hidden layers and different number (K1 and K2) neurons per layer.
4) Support Vector Machines (SVM) utilizing a) linear kernel and b) gaussian kernel
5) Naive Bayes classifier with normal distribution for each element  


Each of the above classification methods must be evaluated with the following evaluation metrics:
1)	a)Accuracy

			Accuracy = (TP+TN)/(P+N)

	b) F1 score
	
			F1score = 2*(Precision*Recall)/(Precision+Recall)

			Precision = TP/(TP+FP)
			Recall = TP/(TP+FN)
			
*where: TP: True Positives, TN: True Negatives, FP: False Positives, FN: False Negatives, P: Positives, N: Negatives*

2) 10-folds cross validation: specifically, the original data will be randomly splited into 10 subsets and for each of these 
subsets the performance of the method (testing set) will be determined by training the rest 9 subsets (training set) with it.

------------------------------------------------------------------------------------------------------------------------------------

Greek presentation of the assignment at "PR2020_Homework1.pdf". 

------------------------------------------------------------------------------------------------------------------------------------
