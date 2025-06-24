import qlib
from qlib.contrib.data.dolphindb_handler import DolphinDBHandler
from qlib.contrib.data.dolphin_dataset import DolphinDBDataset

# 初始化Qlib（最小化）
qlib.init()

# DolphinDB配置

dolphindb_config = {
    "host": "172.16.12.30",      # DolphinDB服务器地址
    "port": 8902,                 # DolphinDB端口
    "username": "admin",          # 用户名
    "password": "123456",         # 密码
    "database": "dfs://wind",       # 数据库名
    "table": "ASHAREEODPRICES",        # 表名
    "start_time": "20231113",
    "end_time": "20241231",
}

try:
    # 创建Handler
    print("创建DolphinDB Handler...")
    handler = DolphinDBHandler(**dolphindb_config)
    
    # 创建Dataset
    print("创建Dataset...")
    dataset = DolphinDBDataset(handler=handler)
    
    # 测试数据获取
    print("测试特征数据...")
    train_features = dataset.prepare("train", col_set="feature")
    print(f"训练特征形状: {train_features.shape}")
    print("特征列:", dataset.get_feature_names())
    
    print("测试标签数据...")
    train_labels = dataset.prepare("train", col_set="label")
    print(f"训练标签形状: {train_labels.shape}")
    print("标签列:", dataset.get_label_names())
    
    print("前5行特征数据:")
    print(train_features.head())
    
    print("测试成功！")
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()