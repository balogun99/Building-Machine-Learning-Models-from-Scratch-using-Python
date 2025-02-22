import numpy as np

class Linear_Regression():
    
    # initiating the parameters
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        
    def fit(self, x, y):    
    # number of training examples & number of features
        # m --> no. of training examples
        # n --> no. of features 
        self.m, self.n = x.shape # number of rows and columns
        
    # initiating the weight and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        
    # implementing the Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()
        
    def update_weights(self):
        y_prediction = self.predict(self.x)
        
        # calculate the gradients
        dw = - (2 * (self.x.T).dot(self.y - y_prediction))/ self.m
        db = - 2 * np.sum(self.y - y_prediction) / self.m
        
        # updating the weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db 
        
    # Line function for prediction
    def predict(self, x):
        return x.dot(self.w) + self.b