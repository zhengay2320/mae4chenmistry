import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import DataConfig
from src.datasets.data import SpectroscopicDataset

logger = logging.getLogger(__name__)

class SpectroscopicCollator:
    """
    Collator for SpectroscopicDataset.
    Handles both dynamic_mixing=True (list of components) and dynamic_mixing=False (single dict).
    Computes mixture spectrum and batches data.
    """
    def __init__(self, max_len: int = DataConfig.MAX_SMILES_LEN):
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of items returned by dataset.__getitem__ (Dicts)
        """
        # Filter out None items
        batch = [x for x in batch if x is not None]
        if not batch:
            logger.warning("Empty batch received in Collator.")
            return {}

        # 1. Stack Mixture Spectra
        mixture_spectra = [x['mixture_spectra'] for x in batch]
        mixture_spectra_tensor = torch.tensor(np.stack(mixture_spectra), dtype=torch.float32)

        # 2. Stack Components and Weights
        # They are already padded to max_k in __getitem__ for "that sample". 
        # But wait, max_k is constant for the dataset (config.MAX_K). 
        # If dataset pads to max_k, then all samples have same K dimension.
        # So we can simply stack.

        component_spectra = [x['component_spectra'] for x in batch]
        component_spectra_tensor = torch.tensor(np.stack(component_spectra), dtype=torch.float32)

        component_weights = [x['component_weights'] for x in batch]
        weights_tensor = torch.tensor(np.stack(component_weights), dtype=torch.float32)

        is_anchor = [x['is_anchor'] for x in batch]
        is_anchor_tensor = torch.tensor(np.stack(is_anchor), dtype=torch.bool)
        
        num_components = [x['num_components'] for x in batch]
        num_components_tensor = torch.tensor(num_components, dtype=torch.long)
        
        global_ids = [x['global_ids'] for x in batch]
        global_ids_tensor = torch.tensor(np.stack(global_ids), dtype=torch.long)

        # Handle Molecular Formulas (List of List of Strings)
        molecular_formulas = [x['molecular_formulas'] for x in batch]

        # 3. Handle SMILES (Variable Length)
        # batch[i]['smiles'] is a list of K lists of tokens.
        # We need to pad tokens to self.max_len.
        
        batch_size = len(batch)
        max_k = component_spectra_tensor.shape[1]
        
        smiles_tensor = torch.zeros((batch_size, max_k, self.max_len), dtype=torch.long)
        
        for i, sample in enumerate(batch):
            smiles_lists = sample['smiles'] # List of K lists
            for j, tokens in enumerate(smiles_lists):
                # tokens might be list or already padded?
                # dataset.tokenize_smiles pads/truncates to max_len.
                # So it should be fine.
                if len(tokens) > self.max_len:
                    tokens = tokens[:self.max_len]
                elif len(tokens) < self.max_len:
                     # This shouldn't happen if tokenize_smiles is consistent, but safe guard
                     tokens = tokens + [0] * (self.max_len - len(tokens))
                
                smiles_tensor[i, j, :] = torch.tensor(tokens, dtype=torch.long)

        # 4. Create Mask
        component_mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
        for i, n in enumerate(num_components):
            component_mask[i, :n] = True

        return {
            "mixture_spectra": mixture_spectra_tensor,      # (B, Spectrum_Dim)
            "component_spectra": component_spectra_tensor,  # (B, Max_K, Spectrum_Dim)
            "component_weights": weights_tensor,            # (B, Max_K)
            "component_mask": component_mask,               # (B, Max_K)
            "is_anchor": is_anchor_tensor,                  # (B, Max_K)
            "smiles": smiles_tensor,                        # (B, Max_K, Max_Len)
            "molecular_formulas": molecular_formulas,       # List[List[str]] (B, Max_K)
            "num_components": num_components_tensor,        # (B)
            "global_ids": global_ids_tensor                 # (B, Max_K)
        }

def get_dataloader(
    dataset: SpectroscopicDataset, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 4, 
    collate_fn: Optional[SpectroscopicCollator] = None
) -> DataLoader:
    """
    Utility to create a DataLoader with the SpectroscopicCollator.
    """
    if collate_fn is None:
        collate_fn = SpectroscopicCollator()
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Config paths
    DATA_DIR = DataConfig.DATA_DIR
    SIMILARITY_INDEX_PATH = DataConfig.SIMILARITY_INDEX_PATH
    SMILES_VOCAB_PATH = DataConfig.SMILES_VOCAB_PATH

    logger.info("Initializing Dataset...")
    # Initialize dataset
    dataset = SpectroscopicDataset(DATA_DIR, SIMILARITY_INDEX_PATH, SMILES_VOCAB_PATH)
    
    # Create DataLoader
    logger.info("Creating DataLoader (batch_size=128)...")
    loader = get_dataloader(dataset, batch_size=128, shuffle=True)
    
    logger.info("Fetching one batch...")
    for batch in loader:
        print("\n" + "="*30)
        print("       BATCH INSPECTION       ")
        print("="*30)
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key:<20} | Shape: {str(list(value.shape)):<20} | Dtype: {value.dtype}")
            else:
                print(f"{key:<20} | Value: {value}")
        
        print("-" * 30)
        
        # Detailed checks
        weights = batch['component_weights']
        mask = batch['component_mask']
        
        print(f"\nSample 0 - Number of components: {batch['num_components'][0]}")
        print(f"Sample 0 - Weights: {weights[0][mask[0]].tolist()}")
        print(f"Sample 0 - Weights Sum: {weights[0].sum().item():.4f}")
        
        if batch['mixture_spectra'] is not None:
            print(f"Sample 0 - Mixture Spectrum Mean: {batch['mixture_spectra'][0].mean().item():.4f}")
            print(f"Sample 0 - Mixture Spectrum Max:  {batch['mixture_spectra'][0].max().item():.4f}")
            
        print("\nTest Complete.")
        break
