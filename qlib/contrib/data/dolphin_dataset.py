import pandas as pd
from typing import Union, List

class DolphinDBDataset:
    """DolphinDB数据集, 模拟DatasetH接口"""
    
    def __init__(self, handler, segments=None):
        self.handler = handler
        self.segments = segments or {
            "train": None,
            "valid": None, 
            "test": None
        }
        
        # 设置数据
        self.handler.setup_data()
    
    def prepare(self, segment: str, col_set: str = None, data_key: str = None):
        """准备数据"""
        return self.handler.prepare(segment, col_set, data_key)
    
    def get_data(self, segment: str = "train"):
        """获取数据"""
        return self.handler.prepare(segment)
    
    def get_feature_names(self):
        """获取特征名称"""
        return self.handler.get_feature_names()
    
    def get_label_names(self):
        """获取标签名称"""
        return self.handler.get_label_names()