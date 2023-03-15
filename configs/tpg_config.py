from pydantic.dataclasses import dataclass
from enum import Enum
from pydantic import validator

class HpEnum(str, Enum):
    NUMBER_OF_TREES = "numTrees"
    MAX_DEPTH = "maxDepth"
    MAX_BINS = "maxBins"

class SpaceNum(str, Enum):
    START = "start"
    STOP = "stop"
    NUM = "num"

SearchingSpace = dict[SpaceNum, int]

SearchingGrid = dict[HpEnum, SearchingSpace]


@dataclass
class TPGConfig:
    model_name: str
    data_path: str
    nan_threshold: float
    min_feature_count: int
    random_seed: int
    hp_tuning: SearchingGrid

    def __getitem__(self, item):
        return getattr(self, item)
    
    @validator('model_name')
    def model_name_must_be_valid(cls, value):
        if value.strip() == "":
            raise ValueError("Model name can't be empty")
        return value  
    
    @validator('data_path')
    def data_path_must_be_valid(cls, value):
        if value.strip() == "":
            raise ValueError("Data path can't be empty")
        return value  
    
    @validator('nan_threshold')
    def threshold_must_be_valid(cls, value):
        if value > 1.0 or value < 0.0:
            raise ValueError("Invalid nan threshold. Must be between 0 and 1")
        return value  
    
    @validator('min_feature_count')
    def min_feature_count_must_be_valid(cls, value):
        if value < 0:
            raise ValueError("Invalid min feature count. Must be greater than 0")
        return value  