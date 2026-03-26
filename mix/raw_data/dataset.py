import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
import faiss
import json
import concurrent.futures
from tqdm import tqdm

# -------------------------
# 配置
# -------------------------
BASE_DIR = Path(r'multimodal_spectroscopic_dataset')
MAX_POINTS = 2000
if not BASE_DIR.exists():
    raise FileNotFoundError(f'Expected folder not found: {BASE_DIR.resolve()}')

# -------------------------
# 统计文件中的分子数量
# -------------------------
total_molecules = 0
file_counts = []

for i in range(245):
    filename = f'aligned_chunk_{i}.parquet'
    file_path = BASE_DIR / filename
    if not file_path.exists():
        continue
    try:
        import pyarrow.parquet as pq
        count = pq.ParquetFile(file_path).metadata.num_rows
    except:
        count = len(pd.read_parquet(file_path))
    file_counts.append({'file': filename, 'molecules': count})
    total_molecules += count

counts_df = pd.DataFrame(file_counts)
print(f'Total molecules: {total_molecules}')
print(f'Files with zero molecules: {counts_df[counts_df["molecules"]==0]["file"].tolist()}')

# -------------------------
# 加载所有 SMILES
# -------------------------
all_files = [BASE_DIR / f'aligned_chunk_{i}.parquet' for i in range(245)]

def load_smiles(path):
    if path.exists():
        return pd.read_parquet(path, columns=['smiles'])
    return None

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(load_smiles, all_files), total=len(all_files)))

dfs = [r for r in results if r is not None]
combined_df = pd.concat(dfs, ignore_index=True)
smiles = combined_df['smiles'].tolist()
ids = combined_df.index.tolist()
print(f"Loaded {len(smiles)} molecules.")

# -------------------------
# 计算 Morgan 指纹
# -------------------------
def compute_fps_chunk(args):
    start_idx, chunk_smiles = args
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    chunk_results = []
    for i, s in enumerate(chunk_smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fp = mfgen.GetFingerprint(m)
            chunk_results.append((start_idx+i, fp))
    return chunk_results

chunk_size = 10000
chunks = [(i, smiles[i:i+chunk_size]) for i in range(0, len(smiles), chunk_size)]
fps, valid_indices = [], []

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(compute_fps_chunk, chunks), total=len(chunks)))
for chunk_res in results:
    for idx, fp in chunk_res:
        valid_indices.append(idx)
        fps.append(fp)

valid_ids = [ids[i] for i in valid_indices]

# -------------------------
# 转换为 Faiss 二进制矩阵
# -------------------------
def fps_to_packed(chunk_fps):
    n = len(chunk_fps)
    arr = np.zeros((n, 2048), dtype=np.uint8)
    for i, fp in enumerate(chunk_fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return np.packbits(arr, axis=1)

fps_chunks = [fps[i:i+chunk_size] for i in range(0, len(fps), chunk_size)]
with concurrent.futures.ProcessPoolExecutor() as executor:
    packed_chunks = list(tqdm(executor.map(fps_to_packed, fps_chunks), total=len(fps_chunks)))
packed_fps = np.vstack(packed_chunks)

# -------------------------
# 构建 Faiss 索引
# -------------------------
faiss.omp_set_num_threads(1)
index = faiss.IndexBinaryFlat(2048)
index.add(packed_fps)
print(f"Faiss index built with {index.ntotal} vectors.")

# -------------------------
# 相似度搜索并重排序
# -------------------------
k_search, k_final, batch_size = 100, 20, 1000

def search_batch(batch):
    start, end = batch
    query_batch = packed_fps[start:end]
    D, I = index.search(query_batch, k_search)
    batch_results = {}
    for i in range(len(query_batch)):
        q_idx = start+i
        q_id = valid_ids[q_idx]
        candidates = [valid_ids[c] for c in I[i] if c != -1]
        candidate_fps = [fps[c] for c in I[i] if c != -1]
        sims = DataStructs.BulkTanimotoSimilarity(fps[q_idx], candidate_fps)
        sorted_candidates = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
        final_neighbors, seen = [], set()
        for cid, sim in sorted_candidates:
            if cid != q_id and cid not in seen:
                final_neighbors.append(cid)
                seen.add(cid)
            if len(final_neighbors) >= k_final:
                break
        batch_results[q_id] = final_neighbors
    return batch_results

batches = [(b, min(b+batch_size, len(packed_fps))) for b in range(0, len(packed_fps), batch_size)]
similarity_map = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(search_batch, batches), total=len(batches)))
for res in results:
    similarity_map.update(res)

with open('similarity_index.json', 'w') as f:
    json.dump(similarity_map, f)
print("Saved similarity index to similarity_index.json")

# -------------------------
# SMILES 字符级 Tokenization
# -------------------------
all_smiles = combined_df['smiles'].tolist()

def unique_chars(smiles_chunk):
    chars = set()
    for s in smiles_chunk:
        chars.update(s)
    return chars

chunks = [all_smiles[i:i+100000] for i in range(0, len(all_smiles), 100000)]
unique_set = set()
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(unique_chars, chunks))
for res in results:
    unique_set.update(res)

sorted_chars = sorted(list(unique_set))
specials = ['<pad>','<start>','<end>','<unk>']
vocab = specials + sorted_chars
stoi = {c:i for i,c in enumerate(vocab)}
itos = {i:c for i,c in enumerate(vocab)}

def tokenize(smiles, max_len=None):
    tokens = [stoi['<start>']]
    tokens += [stoi.get(c, stoi['<unk>']) for c in smiles]
    tokens.append(stoi['<end>'])
    if max_len:
        tokens = tokens[:max_len] + [stoi['<pad>']]*(max_len-len(tokens))
    return tokens

def detokenize(token_ids):
    return ''.join([itos[tid] for tid in token_ids if tid not in [stoi['<pad>'], stoi['<start>'], stoi['<end>']]])

with open('smiles_vocab.json','w') as f:
    json.dump({'stoi':stoi,'vocab':vocab}, f)
print("Saved vocabulary to smiles_vocab.json")

# -------------------------
# 光谱可视化 (第一个分子)
# -------------------------
first_file = BASE_DIR / 'aligned_chunk_0.parquet'
if first_file.exists():
    df_full = pd.read_parquet(first_file)
    mol_data = df_full.iloc[0]
    exclude_cols = ['pos','atomic_numbers','force','dipole_moment','smiles','id']
    potential_spectra = [col for col in df_full.columns if isinstance(mol_data[col], (list, np.ndarray)) and len(mol_data[col])>10 and col not in exclude_cols]

    for col in potential_spectra:
        try:
            y_data = np.array(mol_data[col], dtype=float).flatten()
            x_data = None
            for candidate in ['frequencies','wavenumbers','x_axis']:
                if candidate in df_full.columns and len(df_full.iloc[0][candidate])==len(y_data):
                    x_data = np.array(df_full.iloc[0][candidate], dtype=float).flatten()
                    break
            plt.figure(figsize=(10,4))
            plt.plot(x_data if x_data is not None else y_data, y_data if x_data is not None else None)
            plt.xlabel('Index' if x_data is None else candidate)
            plt.ylabel('Intensity')
            plt.title(f"Spectrum: {col}")
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Could not plot {col}: {e}")
