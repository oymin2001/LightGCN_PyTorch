

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import random
import os
import glob

class DataLoader(object):
  def __init__(self, train_df, test_df, path_dir, col_name, seed=None):
    self.col_user = col_name["col_user"]
    self.col_item = col_name["col_item"]
    self.col_rating = col_name["col_rating"]
    self.col_prediction = "prediction"
    self.path_dir = path_dir

    self.train_df, self.test_df = self._data_processing(train_df, test_df) # See Data prerpocessing
    self.interaction_status = self._make_interaction_status()
    self.R = self._make_user_item_interaction()

    random.seed(seed)

  def _data_processing(self, train_df, test_df):
    preprocessed_df_path = self.path_dir["preprocessed_df_path"]

    df = pd.concat([train_df, test_df], axis=0)
    df = df[df[self.col_rating] > 0]

    self.n_users, self.n_items = df[self.col_user].nunique(), df[self.col_item].nunique()
    self.n_users_train, self.n_items_train = train_df[self.col_user].nunique(), train_df[self.col_item].nunique()

    self.user_idx = df[[self.col_user]].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={"index":self.col_user + "_idx"})
    self.item_idx = df[[self.col_item]].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={"index":self.col_item + "_idx"})


    self.user2id = dict( zip(self.user_idx[self.col_user], self.user_idx[self.col_user + "_idx"]))
    self.item2id = dict( zip(self.item_idx[self.col_item], self.item_idx[self.col_item + "_idx"]))

    train_df[self.col_user] = train_df[self.col_user].map(self.user2id)
    train_df[self.col_item] = train_df[self.col_item].map(self.item2id)

    test_df[self.col_user] = test_df[self.col_user].map(self.user2id)
    test_df[self.col_item] = test_df[self.col_item].map(self.item2id)

    os.makedirs(preprocessed_df_path, exist_ok=True)

    train_df.to_parquet(preprocessed_df_path+"train.parquet", engine='pyarrow', compression='gzip')
    test_df.to_parquet(preprocessed_df_path+"test.parquet", engine='pyarrow', compression='gzip')

    return train_df, test_df

  def _make_interaction_status(self):
    return self.train_df.groupby(self.col_user)[self.col_item].apply(set).reset_index().rename(columns={self.col_item : self.col_item+"_interacted"})

  def _make_user_item_interaction(self):
    R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) # remark. Initialize to train-test dataset
    R[self.train_df[self.col_user], self.train_df[self.col_item]] = 1.0 # store the information about train data
    return R # stored elements in Dictionary Of Keys format

  def create_norm_adj_mat(self, is_exists=False, chunk_size=10000, sub_chunk_size=1000):
    file_path= self.path_dir["norm_adj_mat_path"]

    if is_exists:
        # Load existing normalized adjacency matrix
        norm_adj_mat = sp.load_npz(file_path)
    else:
        # Step 1: Construct adjacency matrix
        R_csr = self.R.tocsr()  # Convert R to CSR format for faster operations
        adj_mat = sp.vstack([
            sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), R_csr]),
            sp.hstack([R_csr.T, sp.csr_matrix((self.n_items, self.n_items))])
        ]).tocsr()  # Convert full adjacency matrix to CSR format

        # Step 2: Calculate row sums and inverse square root diagonal values
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()  # Sum rows
        d_inv_sqrt = np.clip(rowsum, 1e-9, None) ** -0.5  # Calculate inverse square root safely

        # Step 3: Initialize empty file-based storage for normalized matrix
        norm_adj_mat_path = self.path_dir["norm_adj_mat_chunk_path"]
        os.makedirs(norm_adj_mat_path, exist_ok=True)

        print("Normalizing adjacency matrix in chunks...")
        for i in tqdm(range(0, adj_mat.shape[0], chunk_size)):
            start, end = i, min(i + chunk_size, adj_mat.shape[0])

            # Process in sub-chunks to reduce memory usage
            chunk_rows = []
            for j in range(start, end, sub_chunk_size):
                sub_start, sub_end = j, min(j + sub_chunk_size, end)
                d_chunk = sp.diags(d_inv_sqrt[sub_start:sub_end])  # Diagonal matrix for the sub-chunk
                normalized_sub_chunk = d_chunk.dot(adj_mat[sub_start:sub_end]).dot(sp.diags(d_inv_sqrt))
                chunk_rows.append(normalized_sub_chunk)

            # Concatenate sub-chunks and save to disk
            normalized_chunk = sp.vstack(chunk_rows).tocsr()
            sp.save_npz(f"{norm_adj_mat_path}/chunk_{i}.npz", normalized_chunk)

        # Step 4: Combine all chunks into the final matrix
        print("Combining chunks into the final normalized adjacency matrix...")
        chunk_files = sorted(glob.glob(f"{norm_adj_mat_path}/chunk_*.npz"))
        norm_adj_mat = sp.vstack([sp.load_npz(f) for f in chunk_files]).tocsr()

        # Step 5: Save final normalized adjacency matrix
        sp.save_npz(file_path, norm_adj_mat)

    return norm_adj_mat


  def train_loader(self, batch_size):
    users = random.sample(range(self.n_users_train), batch_size)
    pos_items = self.interaction_status.iloc[users,1].apply(lambda x: random.choice(list(x)))
    neg_items = self.interaction_status.iloc[users,1].apply(self.sample_neg)

    return np.array(users), np.array(pos_items), np.array(neg_items)

  def sample_neg(self, x):
    if len(x) >= self.n_items: # Given user is interaced to all items i.e. All items are positive samples
        raise ValueError("A user has voted in every item. Can't find a negative sample.")
    while True:
        neg_id = random.randint(0, self.n_items - 1) # choose random number
        if neg_id not in x: # if neg_id is in x, then it is positive sample i.e. sampling again
            return neg_id

