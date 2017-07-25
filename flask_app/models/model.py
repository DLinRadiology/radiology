class Model(object):
    """Base Class for Models"""
    def __init__(self):
        self.image = None

    def predict(self, image):
        raise NotImplementedError('base')
