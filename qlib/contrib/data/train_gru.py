import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Optional, Dict, Any
import os
import matplotlib.pyplot as plt

from qlib.contrib.data.wind_dataset import create_train_val_test_datasets
from qlib.contrib.model.pytorch_gru import GRUModel

torch.set_float32_matmul_precision('medium')

class LightningGRU(pl.LightningModule):
    """PyTorch Lightning GRU模型"""
    
    def __init__(self,
                 d_feat: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 lr: float = 0.001,
                 weight_decay: float = 1e-5,
                 optimizer: str = "adam",
                 scheduler: str = "cosine",
                 loss_fn: str = "mse",
                 **kwargs):
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 模型参数
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer.lower()
        self.scheduler_name = scheduler.lower()
        self.loss_fn_name = loss_fn.lower()
        
        # 初始化模型
        self.model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 损失函数
        if self.loss_fn_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_fn_name == "mae":
            self.loss_fn = nn.L1Loss()
        elif self.loss_fn_name == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}")
        
        # 指标记录
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        features, labels = batch
        
        # 确保标签是1维的
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        # 前向传播
        predictions = self(features)
        loss = self.loss_fn(predictions, labels)
        
        # 计算额外指标
        mae = nn.L1Loss()(predictions, labels)
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, logger=True, sync_dist=True)
        
        # 保存输出用于epoch结束时的额外处理
        self.train_step_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions.detach(),
            'targets': labels.detach()
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        features, labels = batch
        
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        predictions = self(features)
        loss = self.loss_fn(predictions, labels)
        
        # 计算其他指标
        mae = nn.L1Loss()(predictions, labels)
        mse = nn.MSELoss()(predictions, labels)
        
        # 记录指标
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mae', mae, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mse', mse, on_epoch=True, logger=True, sync_dist=True)
        
        # 保存输出
        self.val_step_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions.detach(),
            'targets': labels.detach()
        })
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        features, labels = batch
        
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        
        predictions = self(features)
        loss = self.loss_fn(predictions, labels)
        
        # 计算指标
        mae = nn.L1Loss()(predictions, labels)
        mse = nn.MSELoss()(predictions, labels)
        
        # 记录指标
        self.log('test_loss', loss, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_mae', mae, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_mse', mse, on_epoch=True, logger=True, sync_dist=True)
        
        # 保存输出
        self.test_step_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions.detach(),
            'targets': labels.detach()
        })
        
        return {
            'test_loss': loss,
            'predictions': predictions.cpu(),
            'targets': labels.cpu()
        }
    
    def on_train_epoch_end(self):
        """训练epoch结束时的处理"""
        if len(self.train_step_outputs) > 0:
            # 计算epoch级别的指标
            all_preds = torch.cat([x['predictions'] for x in self.train_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.train_step_outputs])
            
            # 计算相关系数
            if len(all_preds) > 1:
                try:
                    corr = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1]
                    if not torch.isnan(corr):
                        self.log('train_corr', corr, logger=True, sync_dist=True)
                except Exception as e:
                    print(f"计算训练相关系数时出错: {e}")
            
            # 清空输出
            self.train_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的处理"""
        if len(self.val_step_outputs) > 0:
            # 计算epoch级别的指标
            all_preds = torch.cat([x['predictions'] for x in self.val_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.val_step_outputs])
            
            # 计算相关系数
            if len(all_preds) > 1:
                try:
                    corr = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1]
                    if not torch.isnan(corr):
                        self.log('val_corr', corr, logger=True, sync_dist=True)
                except Exception as e:
                    print(f"计算验证相关系数时出错: {e}")
            
            # 记录预测分布到TensorBoard
            if self.logger and hasattr(self.logger, 'experiment'):
                pred_np = all_preds.cpu().numpy()
                target_np = all_targets.cpu().numpy()
                
                try:
                    # 记录散点图
                    fig = self._create_scatter_plot(pred_np, target_np, "Validation: Predictions vs Targets")
                    self.logger.experiment.add_figure('val_predictions_scatter', fig, self.current_epoch)
                    plt.close(fig)  # 关闭图像释放内存
                    
                    # 记录直方图
                    self.logger.experiment.add_histogram('val_predictions', pred_np, self.current_epoch)
                    self.logger.experiment.add_histogram('val_targets', target_np, self.current_epoch)
                    
                except Exception as e:
                    print(f"记录验证图表时出错: {e}")
            
            # 清空输出
            self.val_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """测试epoch结束时的处理"""
        if len(self.test_step_outputs) > 0:
            # 计算epoch级别的指标
            all_preds = torch.cat([x['predictions'] for x in self.test_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.test_step_outputs])
            
            # 计算相关系数
            if len(all_preds) > 1:
                try:
                    corr = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1]
                    if not torch.isnan(corr):
                        self.log('test_corr', corr, logger=True, sync_dist=True)
                except Exception as e:
                    print(f"计算测试相关系数时出错: {e}")
            
            # 记录测试结果到TensorBoard
            if self.logger and hasattr(self.logger, 'experiment'):
                pred_np = all_preds.cpu().numpy()
                target_np = all_targets.cpu().numpy()
                
                try:
                    # 记录散点图
                    fig = self._create_scatter_plot(pred_np, target_np, "Test: Predictions vs Targets")
                    self.logger.experiment.add_figure('test_predictions_scatter', fig, 0)
                    plt.close(fig)
                    
                    # 记录直方图
                    self.logger.experiment.add_histogram('test_predictions', pred_np, 0)
                    self.logger.experiment.add_histogram('test_targets', target_np, 0)
                    
                except Exception as e:
                    print(f"记录测试图表时出错: {e}")
            
            # 清空输出
            self.test_step_outputs.clear()
    
    def _create_scatter_plot(self, predictions, targets, title):
        """创建散点图"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(targets, predictions, alpha=0.5)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(title)
        
        # 计算R²
        try:
            r2 = np.corrcoef(predictions, targets)[0, 1] ** 2
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"计算R²时出错: {e}")
        
        return fig
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        features, _ = batch
        predictions = self(features)
        return predictions.cpu()
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        
        # 优化器
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        # 学习率调度器
        if self.scheduler_name == "cosine":
            scheduler = {
                'scheduler': optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.trainer.max_epochs,
                    eta_min=self.lr * 0.01
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        
        elif self.scheduler_name == "reduce":
            scheduler = {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        
        elif self.scheduler_name == "step":
            scheduler = {
                'scheduler': optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=20,
                    gamma=0.5
                ),
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        
        else:
            return optimizer


class GRUDataModule(pl.LightningDataModule):
    """PyTorch Lightning数据模块"""
    
    def __init__(self,
                 data_csv: Optional[str] = None,
                 data_df: Optional[pd.DataFrame] = None,
                 date_column: str = 'date',
                 symbol_column: str = 'symbol',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 split_method: str = 'time',
                 sequence_length: int = 10,
                 prediction_horizon: int = 1,
                 price_column: str = 'close',
                 normalize_labels: bool = True,
                 use_mad_normalization: bool = True,
                 clip_outliers: bool = True,
                 clip_range: tuple = (-3, 3),
                 batch_size: int = 64,
                 num_workers: int = 4,
                 cache_dir: str = "./cache",
                 use_cache: bool = True,
                 save_processor_path: Optional[str] = None,
                 random_seed: int = 42):
        super().__init__()
        
        # 保存参数
        self.data_csv = data_csv
        self.data_df = data_df
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_method = split_method
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.price_column = price_column
        self.normalize_labels = normalize_labels
        self.use_mad_normalization = use_mad_normalization
        self.clip_outliers = clip_outliers
        self.clip_range = clip_range
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.save_processor_path = save_processor_path
        self.random_seed = random_seed
        
        # 数据集将在setup中创建
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.processor = None
        self.d_feat = None
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if self.train_dataset is None:
            print("创建数据集...")
            
            self.train_dataset, self.val_dataset, self.test_dataset, self.processor = create_train_val_test_datasets(
                data_csv=self.data_csv,
                data_df=self.data_df,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                split_method=self.split_method,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
                price_column=self.price_column,
                normalize_labels=self.normalize_labels,
                use_mad_normalization=self.use_mad_normalization,
                clip_outliers=self.clip_outliers,
                clip_range=self.clip_range,
                cache_dir=self.cache_dir,
                use_cache=self.use_cache,
                save_processor_path=self.save_processor_path,
                random_seed=self.random_seed
            )
            
            # 推断特征维度
            sample_features, _ = self.train_dataset[0]
            self.d_feat = sample_features.shape[-1]
            print(f"特征维度: {self.d_feat}")
    
    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self):
        """预测数据加载器"""
        return self.test_dataloader()


def train_model(config: Dict[str, Any]):
    """训练模型主函数"""
    
    # 设置随机种子
    pl.seed_everything(config.get('seed', 42))
    
    # 创建数据模块
    data_module = GRUDataModule(**config['data'])
    
    # 设置数据（推断特征维度）
    data_module.setup()
    
    # 创建模型
    model = LightningGRU(
        d_feat=data_module.d_feat,
        **config['model']
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建回调函数
    callbacks = []
    
    # 早停
    if config['trainer'].get('early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00001,
            patience=config['trainer'].get('patience', 10),
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['trainer'].get('checkpoint_dir', './checkpoints'),
        filename='gru-{epoch:02d}-{val_loss:.6f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # 创建TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['trainer'].get('log_dir', './logs'),
        name='gru_experiment',
        version=config['trainer'].get('version', None),
        log_graph=False
    )
    
    # 记录超参数到TensorBoard
    tb_logger.log_hyperparams({
        "model_params": sum(p.numel() for p in model.parameters()),
        "d_feat": data_module.d_feat,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_size": len(data_module.train_dataset),
        "val_size": len(data_module.val_dataset),
        "test_size": len(data_module.test_dataset),
        "sequence_length": data_module.sequence_length,
        "prediction_horizon": data_module.prediction_horizon,
        **config['model']
    })
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['trainer'].get('max_epochs', 100),
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        precision=config['trainer'].get('precision', 32),
        deterministic=True,
        enable_progress_bar=True
    )
    
    # 记录模型图
    # try:
    #     sample_input = torch.randn(1, data_module.sequence_length, data_module.d_feat)
    #     tb_logger.experiment.add_graph(model, sample_input)
    # except Exception as e:
    #     print(f"记录模型图时出错: {e}")
    
    # 训练模型
    print("开始训练...")
    trainer.fit(model, data_module)
    

    # 在训练完成后尝试记录模型图（可选）
    try:
        # 现在模型已经绑定到trainer，可以安全记录
        sample_input = torch.randn(1, data_module.sequence_length, data_module.d_feat)
        model.eval()
        with torch.no_grad():
            tb_logger.experiment.add_graph(model, sample_input)
        model.train()
        print("训练后成功记录模型图")
    except Exception as e:
        print(f"训练后记录模型图失败: {e}")
        print("这不影响训练结果")

    # 测试模型
    print("测试模型...")
    test_results = trainer.test(model, data_module)
    
    # 预测
    print("生成预测...")
    predictions = trainer.predict(model, data_module)
    predictions = torch.cat(predictions).numpy()
    
    # 保存预测结果
    pred_df = pd.DataFrame({'prediction': predictions})
    pred_df.to_csv('./predictions.csv', index=False)
    print("预测结果已保存到 predictions.csv")
    
    # 记录最终结果到TensorBoard
    final_metrics = {
        "final_test_loss": test_results[0]['test_loss'],
        "final_test_mae": test_results[0]['test_mae'],
        "final_test_mse": test_results[0]['test_mse'],
        "final_best_val_loss": trainer.checkpoint_callback.best_model_score.item()
    }
    
    for key, value in final_metrics.items():
        tb_logger.experiment.add_scalar(key, value, 0)
    
    return model, trainer, test_results


def main():
    """主函数"""
    
    # 配置参数
    config = {
        'seed': 42,
        
        # 数据配置
        'data': {
            'data_csv': "stock_data.csv",  # 你的数据文件
            'date_column': 'date',
            'symbol_column': 'symbol',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'split_method': 'time',
            'sequence_length': 10,
            'prediction_horizon': 1,
            'price_column': 'close',
            'normalize_labels': True,
            'use_mad_normalization': True,
            'batch_size': 512,
            'num_workers': 4,
            'cache_dir': "./cache",
            'use_cache': True,
            'save_processor_path': './processor_stats.pkl'
        },
        
        # 模型配置
        'model': {
            'hidden_size': 128,
            'num_layers': 4,
            'dropout': 0.2,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'loss_fn': 'mse'
        },
        
        # 训练配置
        'trainer': {
            'max_epochs': 100,
            'early_stopping': True,
            'patience': 15,
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
            'precision': 32,
            'version': None  # 可以指定版本号
        }
    }
    
    try:
        # 训练模型
        model, trainer, test_results = train_model(config)
        
        print("\n=== 训练完成 ===")
        print(f"最佳模型路径: {trainer.checkpoint_callback.best_model_path}")
        print(f"测试结果: {test_results}")
        print(f"TensorBoard日志保存在: {trainer.logger.log_dir}")
        print("运行以下命令查看TensorBoard:")
        print(f"tensorboard --logdir {trainer.logger.log_dir}")
        
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()