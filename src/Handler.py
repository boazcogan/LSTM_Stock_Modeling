class Handler:
    def __init__(self, epochs, loss_method, regularization_method, learning_rate):
        self.epochs = epochs
        self.loss_method = loss_method
        self.regularization_method = regularization_method
        self.learning_rate = learning_rate
        self.model = None

