class Handler:
    def __init__(self, epochs, loss_method, regularization_method, learning_rate, batch_size, l1enable=False, alpha=0.01):
        self.epochs = epochs
        self.loss_method = loss_method
        self.regularization_method = regularization_method
        self.learning_rate = learning_rate
        self.model = None
        self.batch_size = batch_size
        self.l1enable = l1enable
        self.alpha = alpha
