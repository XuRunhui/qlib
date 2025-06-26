import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path

class FinancialDataProcessor:
    """金融数据预处理器,支持MAD标准化和统计信息记录"""
    
    def __init__(self, use_mad_normalization=True, clip_outliers=True, clip_range=(-3, 3)):
        self.use_mad_normalization = use_mad_normalization
        self.clip_outliers = clip_outliers
        self.clip_range = clip_range
        
        # 存储统计信息
        self.feature_stats = {}
        self.is_fitted = False
    
    def mad_normalize(self, x, axis=0):
        """MAD (Median Absolute Deviation) 标准化"""
        median = np.nanmedian(x, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(x - median), axis=axis, keepdims=True) * 1.4826
        # 避免除零
        mad = np.where(mad == 0, 1, mad)
        return (x - median) / mad
    
    def z_score_normalize(self, x, axis=0):
        """标准Z-Score标准化"""
        mean = np.nanmean(x, axis=axis, keepdims=True)
        std = np.nanstd(x, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)
        return (x - mean) / std
    
    def fit(self, df_features, df_labels=None):
        """拟合统计参数"""
        self.feature_names = df_features.columns.tolist()
        
        # 处理特征
        features_array = df_features.values
        
        if self.use_mad_normalization:
            # MAD统计
            self.feature_stats['median'] = np.nanmedian(features_array, axis=0)
            self.feature_stats['mad'] = np.nanmedian(
                np.abs(features_array - self.feature_stats['median']), axis=0
            ) * 1.4826
            # 避免除零
            self.feature_stats['mad'] = np.where(self.feature_stats['mad'] == 0, 1, self.feature_stats['mad'])
        else:
            # 传统统计
            self.feature_stats['mean'] = np.nanmean(features_array, axis=0)
            self.feature_stats['std'] = np.nanstd(features_array, axis=0)
            self.feature_stats['std'] = np.where(self.feature_stats['std'] == 0, 1, self.feature_stats['std'])
        
        # 处理标签（如果有）
        if df_labels is not None:
            labels_array = df_labels.values
            self.label_stats = {
                'mean': np.nanmean(labels_array, axis=0),
                'std': np.nanstd(labels_array, axis=0)
            }
            self.label_stats['std'] = np.where(self.label_stats['std'] == 0, 1, self.label_stats['std'])
        
        self.is_fitted = True
        print(f"拟合完成 - 特征维度: {len(self.feature_names)}")
        
    def transform_features(self, df_features):
        """转换特征"""
        if not self.is_fitted:
            raise ValueError("请先调用fit()方法拟合参数")
        
        features_array = df_features.values.copy()
        
        if self.use_mad_normalization:
            # MAD标准化
            features_array = (features_array - self.feature_stats['median']) / self.feature_stats['mad']
        else:
            # Z-Score标准化
            features_array = (features_array - self.feature_stats['mean']) / self.feature_stats['std']
        
        # 异常值处理
        if self.clip_outliers:
            features_array = np.clip(features_array, self.clip_range[0], self.clip_range[1])
        
        # 处理NaN值
        features_array = np.nan_to_num(features_array, nan=0.0)
        
        return pd.DataFrame(features_array, columns=df_features.columns, index=df_features.index)
    
    def transform_labels(self, df_labels):
        """转换标签（简单标准化）"""
        if not hasattr(self, 'label_stats'):
            return df_labels
        
        labels_array = df_labels.values.copy()
        labels_array = (labels_array - self.label_stats['mean']) / self.label_stats['std']
        labels_array = np.nan_to_num(labels_array, nan=0.0)
        
        return pd.DataFrame(labels_array, columns=df_labels.columns, index=df_labels.index)
    
    def fit_transform(self, df_features, df_labels=None):
        """拟合并转换"""
        self.fit(df_features, df_labels)
        transformed_features = self.transform_features(df_features)
        transformed_labels = self.transform_labels(df_labels) if df_labels is not None else None
        return transformed_features, transformed_labels
    
    def save_stats(self, filepath):
        """保存统计信息"""
        stats_to_save = {
            'feature_stats': self.feature_stats,
            'feature_names': self.feature_names,
            'use_mad_normalization': self.use_mad_normalization,
            'clip_outliers': self.clip_outliers,
            'clip_range': self.clip_range,
            'is_fitted': self.is_fitted
        }
        
        if hasattr(self, 'label_stats'):
            stats_to_save['label_stats'] = self.label_stats
        
        with open(filepath, 'wb') as f:
            pickle.dump(stats_to_save, f)
        print(f"统计信息已保存到: {filepath}")
    
    def load_stats(self, filepath):
        """加载统计信息"""
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
        
        self.feature_stats = stats['feature_stats']
        self.feature_names = stats['feature_names']
        self.use_mad_normalization = stats['use_mad_normalization']
        self.clip_outliers = stats['clip_outliers']
        self.clip_range = stats['clip_range']
        self.is_fitted = stats['is_fitted']
        
        if 'label_stats' in stats:
            self.label_stats = stats['label_stats']
        
        print(f"统计信息已从 {filepath} 加载")


class FinancialDataset(Dataset):
    """金融时序数据集"""
    
    def __init__(self, 
                 features_df,
                 processor=None,
                 sequence_length=20,
                 prediction_horizon=1,
                 return_index=False,
                 price_column='close',
                 normalize_labels=False,
                 is_training=False,
                cache_dir="./cache",  # 新增
                dataset_name="dataset",  # 新增
                use_cache=True):
        """
        Args:
            features_df: 特征DataFrame
            processor: 数据预处理器
            sequence_length: 序列长度
            prediction_horizon: 预测时间间隔（几天后的收益率）
            return_index: 是否返回索引信息
            price_column: 用于计算收益率的价格列名（如'close', 'adjclose'等）
            normalize_labels: 是否对生成的标签进行标准化
            is_training: 是否为训练集（用于数据增强等）
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.return_index = return_index
        self.price_column = price_column
        self.normalize_labels = normalize_labels
        self.is_training = is_training
        # 缓存相关
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.use_cache = use_cache
        # 数据复制
        self.raw_features = features_df.copy()
        
        print(f"数据形状: {self.raw_features.shape}")
        print(f"可用列名: {list(self.raw_features.columns)}")
        
        # 检查价格列是否存在
        if self.price_column not in self.raw_features.columns:
            available_price_cols = [col for col in self.raw_features.columns 
                                  if any(price_name in col.lower() 
                                        for price_name in ['close', 'price', 'adj'])]
            if available_price_cols:
                self.price_column = available_price_cols[0]
                print(f"未找到'{self.price_column}'列，使用'{self.price_column}'列计算收益率")
            else:
                raise ValueError(f"未找到价格列，可用列: {list(self.raw_features.columns)}")
        
        # 计算标签（未来收益率）
        self.labels = self._calculate_labels()
        
        # 数据预处理
        if processor is not None:
            self.features = processor.transform_features(self.raw_features)
            if self.normalize_labels and self.labels is not None:
                # 对标签进行简单的截面标准化
                self.labels = self._normalize_labels(self.labels)
        else:
            self.features = self.raw_features
        
        # 准备序列数据
        self._prepare_sequences()
        
    def _calculate_labels(self):
        """计算未来收益率标签"""
        print(f"使用'{self.price_column}'列计算未来{self.prediction_horizon}天的收益率...")
        
        labels_list = []
        
        # 按股票分组计算收益率
        grouped = self.raw_features.groupby(level=1)  # level=1是symbol
        
        for symbol, group in grouped:
            # 排序确保时间顺序
            group = group.sort_index(level=0)  # level=0是date
            
            # 计算未来收益率
            prices = group[self.price_column]
            future_returns = prices.shift(-self.prediction_horizon) / prices - 1
            
            # 创建标签DataFrame
            label_df = pd.DataFrame({
                'future_return': future_returns
            }, index=group.index)
            
            labels_list.append(label_df)
        
        # 合并所有标签
        all_labels = pd.concat(labels_list)
        all_labels = all_labels.sort_index()
        
        # 删除NaN值
        all_labels = all_labels.dropna()
        
        print(f"生成标签数量: {len(all_labels)}")
        print(f"标签统计: 均值={all_labels['future_return'].mean():.6f}, "
              f"标准差={all_labels['future_return'].std():.6f}")
        
        return all_labels
    
    def _normalize_labels(self, labels):
        """对标签进行截面标准化"""
        print("对标签进行截面标准化...")
        
        normalized_labels = labels.copy()
        
        # 按日期分组进行截面标准化
        def cross_sectional_normalize(group):
            if len(group) > 1:
                mean = group.mean()
                std = group.std()
                if std > 0:
                    return (group - mean) / std
            return group
        
        # 按日期（level=0）分组标准化
        normalized_labels['future_return'] = (
            normalized_labels.groupby(level=0)['future_return']
            .transform(cross_sectional_normalize)
        )
        
        return normalized_labels
        
    def _prepare_sequences(self):
        """准备时序序列数据"""
        cache_filename = f"{self.dataset_name}_seq{self.sequence_length}_pred{self.prediction_horizon}_{self.price_column}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 如果使用缓存且文件存在，直接加载
        if self.use_cache and os.path.exists(cache_path):
            print(f"从缓存加载序列数据: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.sequences = cached_data['sequences']
            print(f"从缓存加载了 {len(self.sequences)} 个序列")
            return
        
        # 否则重新计算序列
        print("重新计算序列数据...")
        self.sequences = []
        
        # 按股票分组
        grouped_features = self.features.groupby(level=1)  # level=1是symbol
        
        for symbol, group_features in grouped_features:
            # 排序确保时间顺序
            group_features = group_features.sort_index(level=0)  # level=0是date
            
            # 获取对应的标签
            try:
                group_labels = self.labels.loc[self.labels.index.get_level_values(1) == symbol]
                group_labels = group_labels.sort_index(level=0)
            except (KeyError, AttributeError):
                print(f"警告: 股票 {symbol} 没有标签数据，跳过")
                continue
            
            # 创建序列
            valid_sequences = 0
            for i in range(len(group_features) - self.sequence_length + 1):
                # 特征序列
                feature_seq = group_features.iloc[i:i+self.sequence_length].values
                
                # 获取序列结束时间对应的标签
                seq_end_date = group_features.index[i+self.sequence_length-1][0]  # 序列最后一天
                
                # 查找对应日期的标签
                try:
                    label_mask = (group_labels.index.get_level_values(0) == seq_end_date)
                    if label_mask.sum() > 0:
                        label = group_labels.loc[label_mask].values[0]  # 取第一个匹配的标签
                    else:
                        continue  # 如果没有对应的标签，跳过这个序列
                except (IndexError, KeyError):
                    continue
                
                # 检查标签是否有效
                if np.isnan(label).any():
                    continue
                
                # 索引信息
                if self.return_index:
                    start_date = group_features.index[i][0]
                    end_date = group_features.index[i+self.sequence_length-1][0]
                    index_info = (symbol, start_date, end_date)
                else:
                    index_info = None
                
                self.sequences.append({
                    'features': feature_seq,
                    'labels': label,
                    'index': index_info
                })
                valid_sequences += 1
            
            if valid_sequences > 0:
                print(f"股票 {symbol}: 生成 {valid_sequences} 个有效序列")
        
        print(f"总共生成 {len(self.sequences)} 个序列，每个序列长度: {self.sequence_length}")

        # 保存到缓存
        if self.use_cache:
            print(f"保存序列数据到缓存: {cache_path}")
            cache_data = {
                'sequences': self.sequences,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'price_column': self.price_column,
                'normalize_labels': self.normalize_labels,
                'dataset_name': self.dataset_name
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)


    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 转换为tensor
        features = torch.FloatTensor(sequence['features'])
        labels = torch.FloatTensor(sequence['labels'])
        
        if self.return_index:
            return features, labels, sequence['index']
        else:
            return features, labels

    def get_feature_names(self):
        """获取特征名称"""
        return self.features.columns.tolist()
    
    def get_label_stats(self):
        """获取标签统计信息"""
        if self.labels is not None:
            return {
                'mean': self.labels['future_return'].mean(),
                'std': self.labels['future_return'].std(),
                'min': self.labels['future_return'].min(),
                'max': self.labels['future_return'].max(),
                'count': len(self.labels)
            }
        return None
    
    def get_stats(self):
        """获取数据集统计信息"""
        stats = {
            'num_sequences': len(self.sequences),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'num_features': self.features.shape[1],
            'feature_names': self.get_feature_names(),
            'date_range': (self.features.index.get_level_values(0).min(), 
                          self.features.index.get_level_values(0).max()),
            'symbols': self.features.index.get_level_values(1).unique().tolist(),
            'price_column_used': self.price_column
        }
        
        # 添加标签统计
        label_stats = self.get_label_stats()
        if label_stats:
            stats['label_stats'] = label_stats
            
        return stats


def create_train_val_test_datasets(
    data_csv=None,
    data_df=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    split_method='time',  # 'time' 或 'random'
    sequence_length=20,
    prediction_horizon=1,
    price_column='close',
    normalize_labels=False,
    use_mad_normalization=True,
    clip_outliers=True,
    clip_range=(-3, 3),
    save_processor_path=None,
    random_seed=42,
    cache_dir="./cache",  # 新增
    use_cache=True):  # 新增
    """
    创建训练、验证、测试数据集
    
    Args:
        data_csv: 数据CSV文件路径
        data_df: 数据DataFrame（可选，替代CSV）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        split_method: 分割方法 ('time' 按时间分割, 'random' 随机分割)
        sequence_length: 序列长度
        prediction_horizon: 预测时间间隔
        price_column: 用于计算收益率的价格列名
        normalize_labels: 是否对标签进行标准化
        use_mad_normalization: 是否使用MAD标准化
        clip_outliers: 是否裁剪异常值
        clip_range: 异常值裁剪范围
        save_processor_path: 保存预处理器的路径
        random_seed: 随机种子
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, processor)
    """
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 加载数据
    if data_df is not None:
        full_data = data_df.copy()
    else:
        full_data = pd.read_csv(data_csv, index_col=[0, 1], parse_dates=[0])
    
    print(f"完整数据形状: {full_data.shape}")
    print(f"数据日期范围: {full_data.index.get_level_values(0).min()} 到 {full_data.index.get_level_values(0).max()}")
    print(f"股票数量: {full_data.index.get_level_values(1).nunique()}")
    
    # 确保比例加起来为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 数据分割
    if split_method == 'time':
        # 按时间分割
        print("按时间顺序分割数据...")
        dates = sorted(full_data.index.get_level_values(0).unique())
        
        train_end_idx = int(len(dates) * train_ratio)
        val_end_idx = int(len(dates) * (train_ratio + val_ratio))
        
        train_end_date = dates[train_end_idx - 1]
        val_end_date = dates[val_end_idx - 1]
        
        print(f"训练集结束日期: {train_end_date}")
        print(f"验证集结束日期: {val_end_date}")
        
        # 分割数据
        train_data = full_data[full_data.index.get_level_values(0) <= train_end_date]
        val_data = full_data[
            (full_data.index.get_level_values(0) > train_end_date) & 
            (full_data.index.get_level_values(0) <= val_end_date)
        ]
        test_data = full_data[full_data.index.get_level_values(0) > val_end_date]
        
    else:
        # 随机分割（按日期随机）
        print("随机分割数据...")
        dates = sorted(full_data.index.get_level_values(0).unique())
        np.random.shuffle(dates)
        
        train_end_idx = int(len(dates) * train_ratio)
        val_end_idx = int(len(dates) * (train_ratio + val_ratio))
        
        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[train_end_idx:val_end_idx])
        test_dates = set(dates[val_end_idx:])
        
        # 分割数据
        train_data = full_data[full_data.index.get_level_values(0).isin(train_dates)]
        val_data = full_data[full_data.index.get_level_values(0).isin(val_dates)]
        test_data = full_data[full_data.index.get_level_values(0).isin(test_dates)]
    
    print(f"训练集形状: {train_data.shape}")
    print(f"验证集形状: {val_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    
    # 创建预处理器并使用训练集拟合
    print("创建预处理器并使用训练集拟合...")
    processor = FinancialDataProcessor(
        use_mad_normalization=use_mad_normalization,
        clip_outliers=clip_outliers,
        clip_range=clip_range
    )
    
    # 只用训练集拟合预处理器
    processor.fit(train_data)
    
    # 保存预处理器
    if save_processor_path:
        processor.save_stats(save_processor_path)
    
    # 创建数据集
    print("创建训练数据集...")
    train_dataset = FinancialDataset(
        features_df=train_data,
        processor=processor,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        price_column=price_column,
        normalize_labels=normalize_labels,
        is_training=True,
        cache_dir=cache_dir,  # 新增
        dataset_name="train",  # 新增
        use_cache=use_cache  # 新增
    )
    
    print("创建验证数据集...")
    val_dataset = FinancialDataset(
        features_df=val_data,
        processor=processor,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        price_column=price_column,
        normalize_labels=normalize_labels,
        is_training=False,
        cache_dir=cache_dir,  # 新增
        dataset_name="valid",  # 新增
        use_cache=use_cache  # 新增
    )
    
    print("创建测试数据集...")
    test_dataset = FinancialDataset(
        features_df=test_data,
        processor=processor,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        price_column=price_column,
        normalize_labels=normalize_labels,
        is_training=False,
        cache_dir=cache_dir,  # 新增
        dataset_name="test",  # 新增
        use_cache=use_cache  # 新增
    )
    
    # 打印统计信息
    print("\n=== 数据集统计信息 ===")
    for name, dataset in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
        stats = dataset.get_stats()
        print(f"\n{name}:")
        print(f"  序列数量: {stats['num_sequences']}")
        print(f"  特征维度: {stats['num_features']}")
        print(f"  日期范围: {stats['date_range'][0]} 到 {stats['date_range'][1]}")
        if 'label_stats' in stats:
            label_stats = stats['label_stats']
            print(f"  标签统计: 均值={label_stats['mean']:.6f}, 标准差={label_stats['std']:.6f}")
    
    return train_dataset, val_dataset, test_dataset, processor


# 使用示例
if __name__ == "__main__":
    # # 创建示例数据进行测试
    # def create_sample_data():
    #     """创建包含价格列的示例数据"""
    #     dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    #     symbols = ['000001.SZ', '000002.SZ', '600000.SH']
        
    #     data = []
    #     for symbol in symbols:
    #         price = 100  # 初始价格
    #         for date in dates:
    #             # 模拟价格随机游走
    #             price *= (1 + np.random.randn() * 0.02)
    #             data.append({
    #                 'date': date,
    #                 'symbol': symbol,
    #                 'open': price * (1 + np.random.randn() * 0.01),
    #                 'high': price * (1 + abs(np.random.randn()) * 0.015),
    #                 'low': price * (1 - abs(np.random.randn()) * 0.015),
    #                 'close': price,
    #                 'volume': np.random.randint(1000000, 10000000),
    #                 'feature1': np.random.randn(),
    #                 'feature2': np.random.randn(),
    #                 'feature3': np.random.randn(),
    #             })
        
    #     df = pd.DataFrame(data)
    #     df.set_index(['date', 'symbol'], inplace=True)
    #     return df
    
    # # 测试
    # print("创建示例数据...")
    # sample_data = create_sample_data()
    # stock_data = pd.read_csv("stock_data.csv", )
    print("创建训练、验证、测试数据集...")
    train_dataset, val_dataset, test_dataset, processor = create_train_val_test_datasets(
        # data_df=stock_data,
        data_csv="stock_data.csv",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_method='time',
        sequence_length=10,
        prediction_horizon=1,
        price_column='close',
        normalize_labels=True,
        use_mad_normalization=True,
        save_processor_path='./processor_stats.pkl',
        cache_dir="./cache",  # 新增
        use_cache=True  # 新增
    )
    
    print("\n创建DataLoader测试...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 测试一个batch
    for features, labels in train_loader:
        print(f"训练集 Batch:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels sample: {labels[:5].squeeze()}")
        break
    
    # 显示预处理器统计信息
    print(f"\n预处理器统计信息:")
    if processor.use_mad_normalization:
        print(f"使用MAD标准化")
        print(f"特征中位数样例: {processor.feature_stats['median'][:5]}")
        print(f"特征MAD样例: {processor.feature_stats['mad'][:5]}")
    else:
        print(f"使用Z-Score标准化")
        print(f"特征均值样例: {processor.feature_stats['mean'][:5]}")
        print(f"特征标准差样例: {processor.feature_stats['std'][:5]}")