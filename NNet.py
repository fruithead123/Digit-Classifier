import numpy as np

class NNet:
    '''Regularized Neural Network with 1 hidden layer'''
    def __init__(self, inLayer, midLayer, outLayer, regu = 0, alpha = 0.1):
        assert inLayer>0 and midLayer>0 and outLayer>0
        self.inLayer = inLayer
        self.midLayer = midLayer
        self.outLayer = outLayer
        self.regu = regu #Regularization parameter
        self.alpha = alpha #Learning rate for gradient descent
        self.initWeights()

    def setRegu(self, newReg):
        self.regu = newReg
    
    def loadWt1(self, w1):
        assert len(w1)==self.midLayer and len(w1[0])==self.inLayer+1
        self.wt1 = w1

    def loadWt2(self, w2):
        assert len(w2)==self.outLayer and len(w2[0]==self.midLayer+1)
        self.wt2 = w2
        
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def sigmoidGrad(self, X):
        return self.sigmoid(X) * (1-self.sigmoid(X))
    
    def initWeights(self):
        '''Randomly initialize weights'''
        ep1 = np.sqrt(6 / (self.inLayer + self.midLayer))
        ep2 = np.sqrt(6 / (self.midLayer + self.outLayer))
        self.wt1 = np.random.rand(self.midLayer, self.inLayer + 1) * 2 * ep1 - ep1
        self.wt2 = np.random.rand(self.outLayer, self.midLayer + 1) * 2 * ep2 - ep2

        #Set bias terms to 1
        for i in range(self.midLayer):
            self.wt1[i][0] = 1
        for i in range(self.outLayer):
            self.wt2[i][0] = 1

    def processLabels(self, y):
        '''Convert labels into label vectors of size outLayer'''
        result = np.zeros([len(y), self.outLayer])
        for i in range(len(y)):
            result[i][y[i]] = 1
        return result

    def computeHypo(self, X): 
        '''Compute the probability vector for each ouput class''' 
        res = self.sigmoid(X.dot(self.wt1.T))
        res = np.concatenate((np.array([1]*len(X))[:, np.newaxis], res), axis = 1)
        return self.sigmoid(res.dot(self.wt2.T))
                             
    
    def computeCost(self, X, y):
        '''Cost function over training set. Labels (y) already processed'''
        m = len(X)
        X = np.concatenate((np.array([1] * m)[:,np.newaxis], X), axis = 1)
        
        cost = 0
        hypo = self.computeHypo(X)

        #cost -= np.trace(y.dot(np.log(hypo.T))) #Taking trace requires massive matrix (m x m)
        #cost -= np.trace((1-y).dot(np.log(1-hypo.T))) 
        for i in range(m):
            for j in range(self.outLayer):
                cost -= y[i][j] * np.log(hypo[i][j])
                cost -= (1-y[i][j]) * np.log(1 - hypo[i][j])
                
        cost /= m

        reg = sum(sum(self.wt1[:, 1:]**2)) + sum(sum(self.wt2[:, 1:]**2))

        return cost + self.regu / (2*m) * reg

    def computeGrads(self, X, y):
        '''Compute connection gradiants via backpropagation. Labels (y) already processed'''
        grad1 = np.zeros([self.midLayer, self.inLayer + 1])
        grad2 = np.zeros([self.outLayer, self.midLayer + 1])
        m = len(X)
        X = np.concatenate((np.array([1] * m)[:,np.newaxis], X), axis = 1)

        #backpropagation
        for i in range(m):
            a1 = X[i,:][np.newaxis].T
            z2 = self.wt1.dot(a1)
            a2 = np.concatenate((np.array([1])[:, np.newaxis], self.sigmoid(z2)), axis = 0)
            z3 = self.wt2.dot(a2)
            a3 = self.sigmoid(z3)

            delta3 = a3 - y[i,:][np.newaxis].T
            delta2 = self.wt2.T.dot(delta3) * self.sigmoidGrad(np.concatenate((np.array([1])[:, np.newaxis], z2), axis = 0))
            delta2 = delta2[1:]

            grad1 += delta2.dot(a1.T)
            grad2 += delta3.dot(a2.T)

        grad1 /= m
        grad2 /= m

        #Add regularization, but not for bias terms
        for i in range(len(grad1)):
            for j in range(len(grad1[0])):
                if j != 0:
                    grad1[i][j] += self.regu / m * self.wt1[i][j]

        for i in range(len(grad2)):
            for j in range(len(grad2[0])):
                if j != 0:
                    grad2[i][j] += self.regu / m * self.wt2[i][j]
        return (grad1, grad2)


    def train(self, examples, labels, num_cycles = 100, trainMode = 0): #*
        '''Fits training set with gradiant descent. Examples stored as rows'''
        #trainMode: 0 - normal, 1 - batch (sqrt n), 2 -Stochastic
        m = len(examples)
        root = int(m**0.5)
        counter = 0
        print("Cycle 0, Iteration 0: %.4f"%(self.computeCost(examples, labels)))
        for i in range(num_cycles):
            if trainMode == 0:
                grad1, grad2 = self.computeGrads(examples, labels)
                self.wt1 -= self.alpha * grad1
                self.wt2 -= self.alpha * grad2
                counter+=1
                 
            elif trainMode == 1:
                for j in range(root):
                    grad1, grad2 = self.computeGrads(examples[root*j : root*j + root], labels[root*j : root*j + root])
                    self.wt1 -= self.alpha * grad1
                    self.wt2 -= self.alpha * grad2
                    counter+=1
                grad1, grad2 = self.computeGrads(examples[root*root:], labels[root*root:])
                self.wt1 -= self.alpha * grad1
                self.wt2 -= self.alpha * grad2
                counter+=1
            else:
                for j in range(m):
                    grad1, grad2 = self.computeGrads(examples[j][np.newaxis], labels[j][np.newaxis])
                    self.wt1 -= self.alpha * grad1
                    self.wt2 -= self.alpha * grad2
                    counter+=1
                    if counter % root == 0:
                        print("Cycle %d, Iteration %d: %.4f"%(i+1, counter, self.computeCost(examples, labels)))
            print("Cycle %d, Iteration %d: %.4f"%(i+1, counter, self.computeCost(examples, labels)))

            

    def predict(self, x):
        '''Compute Hypothesis by forward propagation'''
        x = np.concatenate((np.array([1])[np.newaxis], np.array(x)[np.newaxis]), axis = 1)
        return self.computeHypo(x)


