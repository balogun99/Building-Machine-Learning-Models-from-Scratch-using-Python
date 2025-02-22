import numpy as np

class Logistic_Regression():
    
    # define the parameters (learning rate & no. of iterations)
    def __init__(self, learning_rate, no_of_iterations):
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
    
    # fit function to train the model in the dataset  
    def fit(self, x, y):
        
    # number of data points in the dataset (number of rows) --> m
    # number of input features in the dataset (number of columns) -->
        self.m, self.n = x.shape
        
    # initiating the weight and bias value
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        
    # implementing Gradient Descent for optimization
        for i in range(self.no_of_iterations):
            self.update_weights()
            
    def update_weights(self):
        
    # Y_hat formula (Sigmoid Function)
        y_hat = 1 / (1 + np.exp(-(self.x.dot(self.w) + self.b))) # wX + b
        
    # find the derivatives or gradient
        dw = (1/self.m)*np.dot(self.x.T, (y_hat - self.y))
        db = (1/self.m)*np.sum(y_hat - self.y)
        
    # updating the weights and bias under gradient descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        
    # Sigmoid Equation and Decision Boundary
                
    def predict(self, x):
        y_pred = 1 / (1 + np.exp(- (x.dot(self.w) + self.b))) # wX + b
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred