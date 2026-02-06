class NeuralOptimizer:
    def __init__(self, model):
        self.model = model

    def quantize(self, bits=8):
        print(f'Quantizing model to {bits} bits...')