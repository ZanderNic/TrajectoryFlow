# std lib imports

# 3-party import

# projekt imports



class Dataset:
    def __init__(self, data, sample_info=None, feature_info=None, name=None):
        self.data = data                        

        self.sample_info = sample_info        
        self.feature_info = feature_info   

        self.name = name
        
    
    def summary(self):
        return {
            "name": self.name,
            "shape": self.shape,
            "n_samples": self.n_samples,
            "n_features": self.n_features
        }
        
    def __str__(self):
        return str(self.summary())
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def n_samples(self):
        return self.data.shape[0]

    @property
    def n_features(self):
        return self.data.shape[1]
    
    
    