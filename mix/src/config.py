import os
import sys
import logging

# Define project root relative to this file (multispectra/src/config.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class GlobalConfig:
    """全局通用配置"""
    SEED = 42

class DataConfig:
    """
    数据生成与处理配置类
    
    用于管理光谱数据的加载、预处理以及在线动态混合生成的各项参数。
    """
    
    # =========================================================================
    # 1. 基础路径配置 (Base Paths)
    # =========================================================================
    # 原始 Parquet 数据目录
    DATA_DIR = "/data/gaohanyu/molecule/raw_data/multimodal_spectroscopic_dataset"
    # 相似性索引文件，用于 Hard Mining (K=Key_ID, V=[Neighbor_IDs])
    SIMILARITY_INDEX_PATH = "/data/gaohanyu/molecule/raw_data/similarity_index.json"
    # SMILES 词表文件
    SMILES_VOCAB_PATH = "/data/gaohanyu/molecule/raw_data/smiles_vocab.json"
    # HuggingFace Dataset 缓存目录
    CACHE_DIR = "/data/gaohanyu/molecule/cache"

    # =========================================================================
    # 2. 数据划分与预处理 (Split & Preprocessing)
    # =========================================================================
    # 数据集划分比例 (必须和为 1.0)
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # SMILES Tokenization 最大长度
    MAX_SMILES_LEN = 128

    # =========================================================================
    # 3. 动态混合生成策略 (Dynamic Mixing Strategy)
    # =========================================================================
    
    # 是否启用在线动态混合生成
    # True: 在 __getitem__ 时实时从单组分数据生成混合谱图
    # False: 直接读取原始数据（用于单组分训练或测试）
    DYNAMIC_MIXING = True
    
    # --- 3.1 组分数量 (K) 配置 ---
    
    # 混合物中组分数量的最小值和最大值 (闭区间 [MIN_K, MAX_K])
    MIN_K = 2
    MAX_K = 6
    
    # K 值的采样策略
    # "uniform": 在 [MIN_K, MAX_K] 范围内均匀采样
    # "weighted": 根据 K_WEIGHTS 指定的概率分布采样
    K_DISTRIBUTION = "weighted"
    
    # 当 K_DISTRIBUTION="weighted" 时的概率分布
    # 长度必须等于 MAX_K - MIN_K + 1
    # 对应顺序: [P(k=MIN_K), ..., P(k=MAX_K)]
    K_WEIGHTS = [0.4, 0.3, 0.2, 0.05, 0.05]

    # --- 3.2 组分选择策略 (Hard Mining) ---
    
    # Hard Mining 概率
    # 以此概率选择与 Anchor (主要组分) 相似的样本作为干扰项
    # 目的: 增加分离难度，迫使模型学习细粒度特征
    HARD_MINING_PROB = 0.2 
    
    # --- 3.3 权重分配策略 (Weight Distribution) ---
    
    # 混合系数的生成策略
    # "equal": 所有组分权重相等 (1/k)
    # "random": 使用 Dirichlet 分布随机生成，模拟真实场景的不平衡混合
    # "manual": 使用 MANUAL_WEIGHTS 指定的固定权重 (此时 k 固定为列表长度)
    WEIGHT_DISTRIBUTION = "random"
    
    # 当 WEIGHT_DISTRIBUTION="manual" 时的固定权重
    # 注意: 启用此项会覆盖 MIN_K/MAX_K 设置
    MANUAL_WEIGHTS = [0.3, 0.3, 0.2, 0.1, 0.1] 

    # 当 WEIGHT_DISTRIBUTION="random" 时的配置
    MIN_WEIGHT = 0.1        # 单个组分的最小权重阈值
    DIRICHLET_ALPHA = 1.0   # Dirichlet 分布参数 (alpha越大越均匀，alpha<1 倾向于稀疏)

    @classmethod
    def setup_logging(cls):
        """
        Setup basic logging configuration.
        """
        logging.basicConfig(
            level=logging.WARNING, # Changed from INFO to WARNING to reduce noise
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    @classmethod
    def validate(cls):
        """
        验证配置参数的合法性
        """
        cls.setup_logging()
        logger = logging.getLogger("DataConfig")
        errors = []
        
        # 1. 验证 Split Ratios
        if abs(cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO - 1.0) > 1e-6:
            errors.append(f"Split ratios must sum to 1.0, got {cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO}")
            
        # 2. 验证 K Range
        if cls.MIN_K > cls.MAX_K:
            errors.append(f"MIN_K ({cls.MIN_K}) cannot be greater than MAX_K ({cls.MAX_K})")
    
        # 3. 验证 K Weights
        if cls.K_DISTRIBUTION == "weighted":
            expected_len = cls.MAX_K - cls.MIN_K + 1
            if len(cls.K_WEIGHTS) != expected_len:
                errors.append(f"K_WEIGHTS length ({len(cls.K_WEIGHTS)}) matches range [{cls.MIN_K}, {cls.MAX_K}] (expected {expected_len})")
            elif abs(sum(cls.K_WEIGHTS) - 1.0) > 1e-4:
                logger.warning(f"K_WEIGHTS sum to {sum(cls.K_WEIGHTS)}, expected 1.0. (Will be normalized automatically)")

        # 4. 验证 Manual Weights
        if cls.WEIGHT_DISTRIBUTION == "manual":
                if abs(sum(cls.MANUAL_WEIGHTS) - 1.0) > 1e-4:
                    logger.warning(f"MANUAL_WEIGHTS sum to {sum(cls.MANUAL_WEIGHTS)}, expected 1.0. (Will be normalized automatically)")

        if errors:
            error_msg = "Configuration Validation Failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # logger.info("Configuration validated successfully.") # Suppress success message

if __name__ == "__main__":
    # 运行此文件进行自检
    DataConfig.validate()
