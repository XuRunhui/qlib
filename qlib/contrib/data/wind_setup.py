import torch
from torch.utils.data import DataLoader
import pandas as pd


# 用于生产环境的数据集创建函数
def create_production_dataset(data_path, stats_path=None):
    """生产环境数据集创建"""
    
    # 如果有预训练的统计信息，直接加载
    if stats_path and os.path.exists(stats_path):
        processor = FinancialDataProcessor()
        processor.load_stats(stats_path)
    else:
        # 否则使用训练数据拟合
        processor = FinancialDataProcessor(use_mad_normalization=True)
        train_data = pd.read_csv(f"{data_path}/train_features.csv", index_col=[0, 1], parse_dates=[0])
        processor.fit(train_data)
        if stats_path:
            processor.save_stats(stats_path)
    
    # 创建数据集
    dataset = FinancialDataset(
        features_csv=f"{data_path}/features.csv",
        labels_csv=f"{data_path}/labels.csv",
        processor=processor,
        sequence_length=20
    )
    
    return dataset, processor

# 数据增强版本（可选）
class AugmentedFinancialDataset(FinancialDataset):
    """带数据增强的金融数据集"""
    
    def __init__(self, *args, noise_std=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_std = noise_std
    
    def __getitem__(self, idx):
        features, labels = super().__getitem__(idx)
        
        # 添加噪声（数据增强）
        if self.training:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
        
        return features, labels


# 1. 假设你已经有了CSV数据
def create_sample_data():
    """创建示例数据"""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    symbols = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
                'feature3': np.random.randn(),
                'return_next': np.random.randn() * 0.02  # 标签
            })
    
    df = pd.DataFrame(data)
    df.set_index(['date', 'symbol'], inplace=True)
    
    # 分离特征和标签
    features = df[['feature1', 'feature2', 'feature3']]
    labels = df[['return_next']]
    
    return features, labels

# 2. 数据处理管道
def create_datasets(train_features_csv, train_labels_csv, 
                   valid_features_csv, valid_labels_csv,
                   test_features_csv, test_labels_csv):
    """创建训练、验证、测试数据集"""
    
    # 初始化预处理器
    processor = FinancialDataProcessor(
        use_mad_normalization=True,  # 使用MAD标准化
        clip_outliers=True,
        clip_range=(-3, 3)
    )
    
    # 加载训练数据并拟合预处理器
    train_features = pd.read_csv(train_features_csv, index_col=[0, 1], parse_dates=[0])
    train_labels = pd.read_csv(train_labels_csv, index_col=[0, 1], parse_dates=[0])
    
    # 拟合预处理器
    processor.fit(train_features, train_labels)
    
    # 保存预处理器参数
    processor.save_stats('./data/processor_stats.pkl')
    
    # 创建数据集
    train_dataset = FinancialDataset(
        features_df=train_features,
        labels_df=train_labels,
        processor=processor,
        sequence_length=20,
        prediction_horizon=1
    )
    
    valid_dataset = FinancialDataset(
        features_csv=valid_features_csv,
        labels_csv=valid_labels_csv,
        processor=processor,
        sequence_length=20,
        prediction_horizon=1
    )
    
    test_dataset = FinancialDataset(
        features_csv=test_features_csv,
        labels_csv=test_labels_csv,
        processor=processor,
        sequence_length=20,
        prediction_horizon=1
    )
    
    return train_dataset, valid_dataset, test_dataset, processor

# 3. 使用示例
if __name__ == "__main__":
    # 创建示例数据
    features, labels = create_sample_data()
    
    # 分割数据
    n_train = int(len(features) * 0.7)
    n_valid = int(len(features) * 0.15)
    
    train_features = features.iloc[:n_train]
    train_labels = labels.iloc[:n_train]
    valid_features = features.iloc[n_train:n_train+n_valid]
    valid_labels = labels.iloc[n_train:n_train+n_valid]
    test_features = features.iloc[n_train+n_valid:]
    test_labels = labels.iloc[n_train+n_valid:]
    
    # 创建预处理器
    processor = FinancialDataProcessor(use_mad_normalization=True)
    
    # 拟合和转换
    processor.fit(train_features, train_labels)
    
    # 显示统计信息
    print("特征统计信息:")
    if processor.use_mad_normalization:
        print(f"Median: {processor.feature_stats['median']}")
        print(f"MAD: {processor.feature_stats['mad']}")
    else:
        print(f"Mean: {processor.feature_stats['mean']}")
        print(f"Std: {processor.feature_stats['std']}")
    
    # 创建数据集
    train_dataset = FinancialDataset(
        features_df=train_features,
        labels_df=train_labels,
        processor=processor,
        sequence_length=10,
        prediction_horizon=1
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    
    # 测试数据加载
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Features shape: {features.shape}")  # [batch_size, sequence_length, num_features]
        print(f"Labels shape: {labels.shape}")      # [batch_size, num_labels]
        
        if batch_idx >= 2:  # 只显示前3个batch
            break
    
    # 保存预处理器
    processor.save_stats('./processor_stats.pkl')
    
    # 显示数据集统计
    print("\n数据集统计:")
    stats = train_dataset.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


