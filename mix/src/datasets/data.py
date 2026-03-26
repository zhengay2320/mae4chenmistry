import os
import glob
import re
import json
import sys
import numpy as np
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import load_dataset, Dataset
import torch
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import DataConfig, GlobalConfig

# Setup logger
logger = logging.getLogger(__name__)

class SpectroscopicDataset:
    """
    Dataset class for multimodal spectroscopic data with online dynamic mixing capabilities.
    """
    def __init__(
        self, 
        data_dir: str, 
        similarity_index_path: str, 
        smiles_vocab_path: str, 
        indices: Optional[Union[List[int], np.ndarray]] = None, 
        preloaded_dataset: Optional[Dataset] = None, 
        min_k: int = DataConfig.MIN_K, 
        max_k: int = DataConfig.MAX_K, 
        dynamic_mixing: bool = DataConfig.DYNAMIC_MIXING,
        min_weight: float = DataConfig.MIN_WEIGHT, 
        hard_mining_prob: float = DataConfig.HARD_MINING_PROB,
        k_distribution: str = DataConfig.K_DISTRIBUTION, 
        k_weights: Optional[List[float]] = DataConfig.K_WEIGHTS,
        weight_distribution: str = DataConfig.WEIGHT_DISTRIBUTION, 
        manual_weights: Optional[List[float]] = DataConfig.MANUAL_WEIGHTS
    ):
        """
        Initialize the SpectroscopicDataset.
        
        Args:
            data_dir (str): Directory containing the parquet files.
            similarity_index_path (str): Path to the similarity_index.json file.
            smiles_vocab_path (str): Path to the smiles_vocab.json file.
            indices (list or np.array, optional): List of global indices to include in this dataset split.
            preloaded_dataset (Dataset, optional): Preloaded Hugging Face dataset object to avoid reloading.
            min_k (int): Minimum number of components for dynamic mixing.
            max_k (int): Maximum number of components for dynamic mixing.
            dynamic_mixing (bool): Whether to enable online dynamic generation of mixtures.
            min_weight (float): Minimum weight for any component in the mixture.
            hard_mining_prob (float): Probability of selecting a hard negative (neighbor) as context.
            k_distribution (str): Sampling strategy for k ("uniform" or "weighted").
            k_weights (list): Weights for k values if k_distribution is "weighted".
            weight_distribution (str): Strategy for weight distribution ("equal", "random", or "manual").
            manual_weights (list): List of weights if weight_distribution is "manual".
        """
        self.data_dir = data_dir
        self.similarity_index_path = similarity_index_path
        self.smiles_vocab_path = smiles_vocab_path
        self.min_k = min_k
        self.max_k = max_k
        self.dynamic_mixing = dynamic_mixing
        self.min_weight = min_weight
        self.hard_mining_prob = hard_mining_prob
        self.k_distribution = k_distribution
        self.k_weights = k_weights
        self.weight_distribution = weight_distribution
        self.manual_weights = manual_weights
        
        # Validate manual weights
        if self.weight_distribution == "manual":
            if not self.manual_weights:
                raise ValueError("Manual weights must be provided when weight_distribution is 'manual'.")
            if abs(sum(self.manual_weights) - 1.0) > 1e-6:
                logger.warning(f"Manual weights sum to {sum(self.manual_weights)}, expected 1.0. Normalizing.")
                total = sum(self.manual_weights)
                self.manual_weights = [w / total for w in self.manual_weights]
        
        # Validate and normalize weights if weighted distribution is selected
        if self.dynamic_mixing and self.k_distribution == "weighted":
            expected_len = self.max_k - self.min_k + 1
            if self.k_weights is None or len(self.k_weights) != expected_len:
                logger.warning(f"K_WEIGHTS length ({len(self.k_weights) if self.k_weights else 0}) does not match range [{self.min_k}, {self.max_k}]. Fallback to uniform.")
                self.k_distribution = "uniform"
            else:
                # Normalize weights to sum to 1.0
                total_w = sum(self.k_weights)
                if total_w > 0:
                    self.k_weights = [w / total_w for w in self.k_weights]
                else:
                    logger.warning("Sum of K_WEIGHTS is 0. Fallback to uniform.")
                    self.k_distribution = "uniform"
        
        if preloaded_dataset is None:
            self.sorted_files = self._get_sorted_files()
            ## logger.info(f"Found {len(self.sorted_files)} parquet files.")
            
            # Using Hugging Face datasets for memory-mapped lazy loading
            logger.info("Loading dataset with memory mapping...")
            
            columns = self._get_columns(self.sorted_files[0])
            logger.info(f"Using columns: {columns}")
            
            # Set cache directory
            # Note: Changing cache_dir will trigger a re-processing of the dataset (approx. 1 hour)
            cache_dir = DataConfig.CACHE_DIR
            logger.info(f"Using cache directory: {cache_dir}")
            
            # Use single process to avoid IOPS issues on cloud storage
            self.dataset = load_dataset("parquet", data_files=self.sorted_files, split="train", columns=columns, cache_dir=cache_dir)
            ## logger.info(f"Dataset loaded. Total samples: {len(self.dataset)}")
        else:
            self.dataset = preloaded_dataset
        
        # Handle indices for splitting
        if indices is None:
            self.indices = np.arange(len(self.dataset))
        else:
            self.indices = np.array(indices)
            
        self.allowed_indices_set = set(self.indices.tolist())
        logger.info(f"Dataset initialized with {len(self.indices)} samples.")

        # 3. Similarity Integration
        ## logger.info("Loading similarity index...")
        self.similarity_index = self._load_similarity_index()
        logger.info(f"Similarity index loaded. Keys: {len(self.similarity_index)}")
        
        # 4. SMILES Vocab Construction / Loading
        ## logger.info("Loading SMILES vocabulary...")
        self.smiles_vocab = self._load_smiles_vocab()
        logger.info(f"SMILES vocabulary loaded. Size: {len(self.smiles_vocab)}")

    def _get_sorted_files(self) -> List[str]:
        """
        Get all aligned_chunk_*.parquet files and sort them naturally by the integer in the filename.
        """
        pattern = os.path.join(self.data_dir, "aligned_chunk_*.parquet")
        files = glob.glob(pattern)
        
        def extract_chunk_id(filename: str) -> int:
            # Extract the number k from aligned_chunk_{k}.parquet
            # use os.path.basename to ensure match only the filename
            basename = os.path.basename(filename)
            match = re.search(r'aligned_chunk_(\d+)\.parquet', basename)
            if match:
                return int(match.group(1))
            return -1 
        
        # Sort files based on the extracted integer
        files.sort(key=extract_chunk_id)
        
        return files

    def _get_columns(self, filepath: str) -> List[str]:
        """
        Get the list of columns from a parquet file, excluding index columns.
        Only keep IR, NMR, and SMILES related columns to speed up loading.
        """
        # Hardcode the required columns to avoid loading massive MS data
        # We need SMILES for ID, IR for main task, and NMR as requested.
        required_columns = [
            'smiles',
            'molecular_formula',
            'ir_spectra', 
            'h_nmr_spectra', 'h_nmr_peaks', 
            'c_nmr_spectra', 'c_nmr_peaks'
        ]
        
        # Verify these columns exist in the file schema
        schema = pq.read_schema(filepath)
        available_columns = schema.names
        
        # Only keep columns that actually exist in the parquet file
        final_columns = [col for col in required_columns if col in available_columns]
        
        # Also include any ID/Index columns if they exist (optional but good practice)
        if 'id' in available_columns:
            final_columns.append('id')
            
        return final_columns

    def _load_similarity_index(self) -> Dict[int, List[int]]:
        """
        Load similarity index and convert keys to integers.
        """
        with open(self.similarity_index_path, 'r') as f:
            data = json.load(f)
        # Convert keys to int as requested: {Global_ID: [Top-20_Neighbor_IDs]}
        return {int(k): v for k, v in data.items()}

    def _load_smiles_vocab(self) -> Dict[str, int]:
        """
        Load SMILES vocabulary.
        """
        with open(self.smiles_vocab_path, 'r') as f:
            vocab_data = json.load(f)
        return vocab_data['stoi']

    def tokenize_smiles(self, smiles: str, max_len: int = DataConfig.MAX_SMILES_LEN) -> List[int]:
        """
        Convert SMILES string to Token ID sequence.
        """
        vocab = self.smiles_vocab
        # Simple character-level tokenization based on the vocab provided
        
        token_ids = [vocab.get(char, vocab.get("<unk>")) for char in smiles]
        
        # Truncate or pad
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids += [vocab.get("<pad>")] * (max_len - len(token_ids))
            
        return token_ids

    @staticmethod
    def split_indices(total_size: int, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split indices into train, val, and test sets.
        """
        # Normalize ratios if they don't sum exactly to 1 due to float precision, but here we expect exact sum
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
             raise ValueError("Ratios must sum to 1.0")
        
        indices = np.arange(total_size)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_end = int(total_size * train_ratio)
        val_end = int(total_size * (train_ratio + val_ratio))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return train_indices, val_indices, test_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get an item by index.
        """
        # Map local index to global index
        global_idx = int(self.indices[idx])

        if not self.dynamic_mixing:
            # Lazy load the data row
            row = self.dataset[global_idx]
            
            neighbor_ids = self.similarity_index.get(global_idx, [])
            
            # Filter neighbors to ensure they are in the current split (prevent data leakage)
            neighbor_ids = [nid for nid in neighbor_ids if nid in self.allowed_indices_set]
            
            # Tokenize SMILES
            smiles = row.get('smiles', '')
            tokenized_smiles = self.tokenize_smiles(smiles)
            
            # For single sample, we treat it as mixture of 1
            raw_ir = row.get('ir_spectra')
            if raw_ir is None:
                # Handle error or return empty
                # For now assume valid data
                return None
            
            # Normalize single sample by its max (consistent with mixture of 1)
            arr = np.array(raw_ir, dtype=np.float32)
            scale = np.max(arr) + 1e-8
            norm_ir = arr / scale
            
            return {
                "mixture_spectra": norm_ir,
                "component_spectra": np.expand_dims(norm_ir, 0), # [1, L]
                "component_weights": np.array([1.0], dtype=np.float32),
                "is_anchor": np.array([True], dtype=bool),
                "num_components": 1,
                "smiles": [tokenized_smiles], # List of lists
                "molecular_formulas": [row.get('molecular_formula', '')], # List of strings
                "global_ids": [global_idx]
            }

        # --- Phase 3: Online Dynamic Generation ---
        
        plan = self.get_mixing_plan(global_idx)
        
        raw_spectra = []
        weights = []
        is_anchor = []
        smiles_list = []
        formulas_list = []
        global_ids = []
        
        # Collect raw data
        valid_plan_items = []
        for item in plan:
            idx_val = item['global_id']
            row = self.dataset[idx_val]
            
            raw_ir = row.get('ir_spectra')
            if raw_ir is None:
                continue
                
            arr = np.array(raw_ir, dtype=np.float32)
            
            raw_spectra.append(arr)
            weights.append(item['weight'])
            is_anchor.append(item['is_anchor'])
            
            s = row.get('smiles', '')
            smiles_list.append(self.tokenize_smiles(s))
            
            f = row.get('molecular_formula', '')
            formulas_list.append(f)
            
            global_ids.append(idx_val)
            valid_plan_items.append(item)

        if not raw_spectra:
            # Fallback if everything failed
            return None 

        # Mix
        # Convert to numpy arrays
        raw_spectra_np = np.stack(raw_spectra) # [K, L]
        weights_np = np.array(weights, dtype=np.float32) # [K]
        is_anchor_np = np.array(is_anchor, dtype=bool) # [K]
        
        # mix = sum(spec * weight)
        # Broadcasting weights: [K, 1] * [K, L] -> [K, L] -> sum dim 0 -> [L]
        mixture = np.sum(raw_spectra_np * weights_np[:, None], axis=0)
        
        # Normalize (Max Scaling)
        scale = np.max(mixture) + 1e-8
        
        mixture_norm = mixture / scale
        targets_norm = (raw_spectra_np * weights_np[:, None]) / scale
        
        # Pad to max_k
        current_k = len(targets_norm)
        pad_n = self.max_k - current_k
        
        if pad_n > 0:
            # Padding
            targets_padded = np.pad(targets_norm, ((0, pad_n), (0, 0)), 'constant')
            weights_padded = np.pad(weights_np, (0, pad_n), 'constant')
            is_anchor_padded = np.pad(is_anchor_np, (0, pad_n), 'constant')
            # For SMILES and IDs, we extend lists
            smiles_padded = smiles_list + [self.tokenize_smiles("")] * pad_n # Pad with empty/unk
            formulas_padded = formulas_list + [""] * pad_n
            global_ids_padded = global_ids + [-1] * pad_n
        else:
            targets_padded = targets_norm
            weights_padded = weights_np
            is_anchor_padded = is_anchor_np
            smiles_padded = smiles_list
            formulas_padded = formulas_list
            global_ids_padded = global_ids

        return {
            "mixture_spectra": mixture_norm,          # [L]
            "component_spectra": targets_padded,      # [Max_K, L]
            "component_weights": weights_padded,      # [Max_K]
            "is_anchor": is_anchor_padded,            # [Max_K]
            "num_components": current_k,
            "smiles": smiles_padded,                  # List of K lists
            "molecular_formulas": formulas_padded,    # List of K strings
            "global_ids": global_ids_padded           # List of K ints
        }

    def normalize_spectrum(self, spectrum: Union[List[float], np.ndarray]) -> Optional[np.ndarray]:
        """
        Normalize spectrum to [0, 1] range.
        """
        if spectrum is None:
            return None
        arr = np.array(spectrum, dtype=np.float16)
        max_val = np.max(arr)
        if max_val > 1e-9:
            return arr / max_val
        return arr

    def get_mixing_plan(self, global_idx: int) -> List[Dict[str, Any]]:
        """
        Generate the mixing plan (indices and weights) without loading the data.
        Returns a list of dicts: [{'global_id': int, 'weight': float, 'is_anchor': bool}, ...]
        """
        # 1. Determine number of components k
        if self.weight_distribution == "manual":
            k = len(self.manual_weights)
        elif self.k_distribution == "weighted":
            k_options = list(range(self.min_k, self.max_k + 1))
            k = np.random.choice(k_options, p=self.k_weights)
        else:
            # Uniform sampling
            k = random.randint(self.min_k, self.max_k)
        
        # 2. Select Components
        # Role A: Anchor (Slot 0) - Forced Lock
        anchor_idx = global_idx
        
        # Role B: Context (Slots 1 to k-1)
        # Prepare neighbors for Hard Mining
        anchor_neighbors = self.similarity_index.get(anchor_idx, [])
        # Leakage Filter: Must be in current split (allowed_indices_set)
        valid_neighbors = [nid for nid in anchor_neighbors if nid in self.allowed_indices_set]
        random.shuffle(valid_neighbors) # Shuffle for "without replacement" sampling
        
        component_indices = [anchor_idx]
        
        for _ in range(k - 1):
            # Decision Branch
            # Hard Mining (probability from config)
            is_hard_mining = (random.random() < self.hard_mining_prob)
            
            selected_idx = None
            
            if is_hard_mining:
                if valid_neighbors:
                    selected_idx = valid_neighbors.pop() # Sample without replacement
                else:
                    # Fallback to Random Mode if no neighbors left
                    is_hard_mining = False
            
            if not is_hard_mining:
                # Random Mode (80% probability or fallback)
                # Completely random from allowed indices
                # Optimization: np.random.choice on large array is slow. Use index selection.
                rand_idx = np.random.randint(len(self.indices))
                selected_idx = int(self.indices[rand_idx])
                
            component_indices.append(selected_idx)
        
        # 3. Generate Proportions and Shuffle
        if self.weight_distribution == "equal":
            weights = np.full(k, 1.0 / k)
        elif self.weight_distribution == "manual":
            weights = np.array(self.manual_weights)
        else:
            # Generate weights using Dirichlet distribution
            # alpha参数可调节分布均匀性
            remaining_weight = 1.0 - (k * self.min_weight)
            raw_weights = np.random.dirichlet([DataConfig.DIRICHLET_ALPHA] * k)
            weights = raw_weights * remaining_weight + self.min_weight
            weights = np.round(weights, 4)
            weights[-1] = 1.0 - np.sum(weights[:-1])
        
        # Create plan
        plan = []
        for i, idx_val in enumerate(component_indices):
            plan.append({
                "global_id": int(idx_val),
                "weight": weights[i],
                "is_anchor": (i == 0)
            })
            
        # Shuffle the order of components
        random.shuffle(plan)
        return plan

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    
    # Configuration paths
    DATA_DIR = DataConfig.DATA_DIR
    SIMILARITY_INDEX_PATH = DataConfig.SIMILARITY_INDEX_PATH
    SMILES_VOCAB_PATH = DataConfig.SMILES_VOCAB_PATH
    
    try:
        logger.info("Initializing full dataset to determine size...")
        # Initialize without indices to get the full size
        full_dataset = SpectroscopicDataset(DATA_DIR, SIMILARITY_INDEX_PATH, SMILES_VOCAB_PATH)
        total_size = len(full_dataset)
        
        # 2. Split Indices
        train_indices, val_indices, test_indices = SpectroscopicDataset.split_indices(
            total_size, 
            train_ratio=DataConfig.TRAIN_RATIO, 
            val_ratio=DataConfig.VAL_RATIO, 
            test_ratio=DataConfig.TEST_RATIO,
            seed=GlobalConfig.SEED
        )
        logger.info(f"Train size: {len(train_indices)}")
        logger.info(f"Val size: {len(val_indices)}")
        logger.info(f"Test size: {len(test_indices)}")
        
        # 3. Instantiate Datasets with splits
        # Reuse the underlying dataset object to avoid reloading and re-printing progress bars
        shared_hf_dataset = full_dataset.dataset

        logger.info("Initializing Train Dataset...")
        train_dataset = SpectroscopicDataset(DATA_DIR, SIMILARITY_INDEX_PATH, SMILES_VOCAB_PATH, indices=train_indices, preloaded_dataset=shared_hf_dataset)
        
        logger.info("Initializing Val Dataset...")
        val_dataset = SpectroscopicDataset(DATA_DIR, SIMILARITY_INDEX_PATH, SMILES_VOCAB_PATH, indices=val_indices, preloaded_dataset=shared_hf_dataset)
        
        logger.info("Initializing Test Dataset...")
        test_dataset = SpectroscopicDataset(DATA_DIR, SIMILARITY_INDEX_PATH, SMILES_VOCAB_PATH, indices=test_indices, preloaded_dataset=shared_hf_dataset)
        
        # Verify constraint: Train dataset should only access train indices
        logger.info("Verifying Train Dataset access...")
        if len(train_dataset) > 0:
            sample_idx = 0
            sample = train_dataset[sample_idx]
            # Since __getitem__ logic changed to return tensors directly in online mode,
            # we need to inspect the internal structure or bypass dynamic mixing to check neighbors.
            # But here we just check if it runs.
            logger.info(f"Sample {sample_idx} retrieved successfully.")
        
        logger.info("Verification complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
