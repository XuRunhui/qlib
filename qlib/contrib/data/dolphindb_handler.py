import dolphindb as ddb
import pandas as pd
import numpy as np
from typing import Union, List
from qlib.log import get_module_logger

class DolphinDBHandler:
    """DolphinDB数据处理器 - 独立实现"""
    
    def __init__(
        self,
        host: str,
        port: int,
        username: str = None,
        password: str = None,
        database: str = "stock_db",
        table: str = "stock_data",
        instruments: Union[str, List[str]] = "all",
        start_time: str = "2015-01-01",
        end_time: str = "2023-12-31",
        fit_start_time: str = None,
        fit_end_time: str = None,
        **kwargs
    ):
        self.host = host
        self.port = port  
        self.username = username
        self.password = password
        self.database = database
        self.table = table
        self.load_table = "loadTable('dfs://wind', 'ASHAREEODPRICES')"
        # 初始化logger
        self.logger = get_module_logger(self.__class__.__name__)
        
        # 连接DolphinDB
        self._connect_dolphindb()
        self._get_data_fields()
        # 处理instruments
        if isinstance(instruments, str) and instruments == "all":
            instruments = self._get_all_instruments()
        elif isinstance(instruments, str):
            instruments = self._get_index_constituents(instruments)
        
        # 存储配置
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        
        # 初始化数据存储
        self.data_train = None
        self.data_valid = None  
        self.data_test = None
        self._data_cache = {}
        
        self.logger.info("DolphinDB Handler 初始化完成")
    
    def _connect_dolphindb(self):
        """连接DolphinDB服务器"""
        try:
            self.session = ddb.session()
            self.session.connect(self.host, self.port, self.username, self.password)
            self.logger.info(f"成功连接到DolphinDB: {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"连接DolphinDB失败: {e}")
            raise
    
    def _get_all_instruments(self) -> List[str]:
        """从DolphinDB获取所有股票代码"""
        try:
            script = f"select distinct S_INFO_WINDCODEE from loadTable('dfs://wind', 'ASHAREEODPRICES') order by S_INFO_WINDCODEE"
            result = self.session.run(script)
            
            if hasattr(result, 'values'):
                instruments = result.values.flatten().tolist()
            elif isinstance(result, list):
                instruments = result
            else:
                instruments = list(result)
            
            self.logger.info(f"获取到{len(instruments)}只股票")
            return instruments[:50]  # 限制数量
            
        except Exception as e:
            self.logger.error(f"获取股票列表失败: {e}")
            return ["000001.SZ", "000002.SZ", "600000.SH"]
    
    def _get_column_mapping(self):
        """获取列名映射 - 针对Wind数据格式"""
        # Wind数据库的列名映射
        mapping = {
            'symbol': 'S_INFO_WINDCODEE',      # 股票代码
            'date': 'TRADE_DT',                # 交易日期
            'open': 'S_DQ_OPEN',               # 开盘价
            'high': 'S_DQ_HIGH',               # 最高价
            'low': 'S_DQ_LOW',                 # 最低价
            'close': 'S_DQ_CLOSE',             # 收盘价
            'volume': 'S_DQ_VOLUME',           # 成交量
            'amount': 'S_DQ_AMOUNT',           # 成交额
            'preclose': 'S_DQ_PRECLOSE',       # 昨收价
            'change': 'S_DQ_CHANGE',           # 涨跌额
            'pctchange': 'S_DQ_PCTCHANGE',     # 涨跌幅
            'adjclose': 'S_DQ_ADJCLOSE',       # 复权收盘价
            'adjopen': 'S_DQ_ADJOPEN',         # 复权开盘价
            'adjhigh': 'S_DQ_ADJHIGH',         # 复权最高价
            'adjlow': 'S_DQ_ADJLOW',           # 复权最低价
            'adjfactor': 'S_DQ_ADJFACTOR',     # 复权因子
        }
        
        self.logger.info(f"Wind数据列名映射: {mapping}")
        return mapping



    def _get_index_constituents(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        self.logger.warning(f"指数成分股功能未实现，使用默认股票列表")
        return self._get_all_instruments()
    
    def setup_data(self, enable_cache: bool = True):
        """设置数据"""
        self.logger.info("开始设置数据...")
        
        # 获取数据
        raw_data = self._fetch_data_from_dolphindb()
        
        if raw_data.empty:
            raise ValueError("无法从DolphinDB获取数据")
        
        # 处理数据
        processed_data = self._process_data(raw_data)
        
        # 分割数据
        self._split_data(processed_data)
        
        self.logger.info("数据设置完成")
    
    def _fetch_data_from_dolphindb(self) -> pd.DataFrame:
        """从DolphinDB获取原始数据"""
        try:
            column_mapping = self._get_column_mapping()
        
            # 构建查询的列名（选择我们需要的列）
            select_cols = [
                f"{column_mapping['symbol']} as symbol",
                f"{column_mapping['date']} as date", 
                f"{column_mapping['open']} as open",
                f"{column_mapping['high']} as high",
                f"{column_mapping['low']} as low", 
                f"{column_mapping['close']} as close",
                f"{column_mapping['volume']} as volume",
                f"{column_mapping['amount']} as amount",
                f"{column_mapping['pctchange']} as pctchange",  # 可以直接使用涨跌幅
                f"{column_mapping['adjclose']} as adjclose"     # 复权价格
            ]

            # 构建股票代码字符串
            instruments_str = "'" + "','".join(self.instruments) + "'"
            select_str = ", ".join(select_cols)
            
            # 构建查询语句
            # script = f""" select {select_str} from loadTable('dfs://wind', 'ASHAREEODPRICES') where S_INFO_WINDCODEE in ({instruments_str}) and TRADE_DT between '{self.start_time}' : '{self.end_time}' order by S_INFO_WINDCODEE, TRADE_DT """
            
            # 构建查询语句
            script = f""" select {select_str} from loadTable("{self.database}", "{self.table}") where {column_mapping['symbol']} in ({instruments_str}) and {column_mapping['date']} between '{self.start_time}' : '{self.end_time}' order by {column_mapping['symbol']}, {column_mapping['date']} """
            
            self.logger.info(f"执行DolphinDB查询SQL:\n{script}")
            result = self.session.run(script)
            
            if result is None or len(result) == 0:
                self.logger.warning("DolphinDB查询返回空结果")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(result)
            self.logger.info(f"查询结果列名: {list(df.columns)}")
            
            # 处理日期格式
            if 'date' in df.columns:
                # Wind数据的日期格式通常是YYYYMMDD，需要转换
                df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
                df.set_index(['date', 'symbol'], inplace=True)
                df.sort_index(inplace=True)
            else:
                self.logger.error(f"缺少日期列")
                return pd.DataFrame()
            
            self.logger.info(f"成功获取数据: {len(df)}条记录")
            return df
                
        except Exception as e:
            self.logger.error(f"从DolphinDB获取数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据，计算特征和标签"""
        if df.empty:
            return df
        
        def compute_features(group):
            group = group.sort_index()
            
            # 计算收益率
            group['return'] = group['close'].pct_change()
            group['return_next'] = group['close'].shift(-1) / group['close'] - 1  # 未来收益率
            
            # 技术指标
            group['ma5'] = group['close'].rolling(5).mean()
            group['ma20'] = group['close'].rolling(20).mean()
            group['volume_ratio'] = group['volume'] / group['volume'].rolling(5).mean()
            group['amplitude'] = (group['high'] - group['low']) / group['close']
            
            # 标准化特征
            group['ma5_ratio'] = group['ma5'] / group['close'] - 1
            group['ma20_ratio'] = group['ma20'] / group['close'] - 1
            
            return group
        
        # 按股票分组处理
        df_processed = df.groupby(level=1).apply(compute_features)
        df_processed.index = df_processed.index.droplevel(0)
        
        # 删除包含NaN的行
        df_processed = df_processed.dropna()
        
        return df_processed
    
    def _split_data(self, df: pd.DataFrame):
        """分割训练、验证、测试数据"""
        dates = df.index.get_level_values(0).unique().sort_values()
        
        # 70% 训练，15% 验证，15% 测试
        train_end_idx = int(len(dates) * 0.7)
        valid_end_idx = int(len(dates) * 0.85)
        
        train_end_date = dates[train_end_idx]
        valid_end_date = dates[valid_end_idx]
        
        self.data_train = df[df.index.get_level_values(0) <= train_end_date].copy()
        self.data_valid = df[
            (df.index.get_level_values(0) > train_end_date) & 
            (df.index.get_level_values(0) <= valid_end_date)
        ].copy()
        self.data_test = df[df.index.get_level_values(0) > valid_end_date].copy()
        
        self.logger.info(f"数据分割完成 - 训练:{len(self.data_train)}, 验证:{len(self.data_valid)}, 测试:{len(self.data_test)}")
    
    def prepare(self, segment: str, col_set: str = None, data_key: str = None):
        """准备数据 - 模拟DatasetH的接口"""
        # 获取对应段的数据
        if segment == "train":
            data = self.data_train
        elif segment == "valid":
            data = self.data_valid
        elif segment == "test":
            data = self.data_test
        else:
            raise ValueError(f"未知的数据段: {segment}")
        
        if data is None or data.empty:
            self.logger.warning(f"{segment} 数据为空")
            return pd.DataFrame()
        
        # 根据col_set返回不同的列
        if col_set == "feature":
            # 返回特征列
            feature_cols = ['return', 'ma5_ratio', 'ma20_ratio', 'volume_ratio', 'amplitude']
            available_cols = [col for col in feature_cols if col in data.columns]
            return data[available_cols]
        elif col_set == "label":
            # 返回标签列
            if 'return_next' in data.columns:
                return data[['return_next']]
            else:
                return pd.DataFrame()
        else:
            # 返回所有数据
            return data
    

    def _get_data_fields(self):
        """获取数据名称"""
        script = f"""columnNames({self.load_table})"""
        self.fields = self.session.run(script)

        self.logger.info(f"getting data fields: {self.fields}")


    def get_feature_names(self):
        """获取特征名称"""
        return ['return', 'ma5_ratio', 'ma20_ratio', 'volume_ratio', 'amplitude']
    
    def get_label_names(self):
        """获取标签名称"""
        return ['return_next']
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'session'):
            try:
                self.session.close()
            except:
                pass