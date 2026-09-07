from abc import ABC, abstractmethod

class Preprocessing(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def set_input(self):
        pass

    @abstractmethod
    def process(self):
        pass
    
    @property
    @abstractmethod
    def get_output(self):
        pass