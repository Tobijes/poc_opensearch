class Model:
    model_name = "Default Model"

    def __init__(self):
        pass
    
    def predict(self, name):
        return f"Hello, {name}. Predicted by {self.model_name}"
