
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch

def convert_to_sparse_tensor(dok_mtrx):

    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)
    values = dok_mtrx_coo.data
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = dok_mtrx_coo.shape

    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return dok_mtrx_sparse_tensor

def get_item_hit(col_name, row):
  return list(set(row[col_name['col_item']]).intersection(row['top_relevent_item']))

def get_recall(col_name, row):
  item_hit = get_item_hit(col_name, row)
  return len(item_hit)/len(row[col_name['col_item']])

def get_precision(col_name, row, K):
  item_hit = get_item_hit(col_name, row)
  return len(item_hit)/K

def get_hit_list(col_name, row):
  return [1 if x in set(row[col_name['col_item']]) else 0 for x in row['top_relevent_item']]

def get_dcg_idcg(col_name, row):
  hit_list = get_hit_list(col_name, row)
  idcg  = sum([1 / np.log1p(idx+1) for idx in range(min(len(row[col_name['col_item']]),len(hit_list)))])
  dcg =  sum([hit / np.log1p(idx+1) for idx, hit in enumerate(hit_list)])
  return dcg / idcg

def get_map(col_name, row):
    item_id_idx = row[col_name['col_item']]
    hit_list = get_hit_list(col_name, row)
    hit_list_cumsum = np.cumsum(hit_list)

    return sum([hit_cumsum*hit/(idx+1) for idx, (hit, hit_cumsum) in enumerate(zip(hit_list, hit_list_cumsum))])/len(item_id_idx)

def get_metrics_with_chunks(col_name, topk_relevance_indices, start, test_interactions, K):
  top_k_relevance_indicies_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
  top_k_relevance_indicies_df[col_name['col_user']] = top_k_relevance_indicies_df.index + start

  top_k_relevance_indicies_df['top_relevent_item'] = top_k_relevance_indicies_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
  top_k_relevance_indicies_df = top_k_relevance_indicies_df[[col_name['col_user'],'top_relevent_item']]

  metrics_df = pd.merge(test_interactions,top_k_relevance_indicies_df, how='inner', on = col_name['col_user'])

  metrics_df['recall'] = metrics_df.apply(lambda x : get_recall(col_name, x), axis = 1) 
  metrics_df['precision'] = metrics_df.apply(lambda x : get_precision(col_name, x, K=K), axis = 1)
  metrics_df['ndcg'] = metrics_df.apply(lambda x : get_dcg_idcg(col_name, x), axis = 1)
  metrics_df['map'] = metrics_df.apply(lambda x : get_map(col_name, x), axis = 1)
  
  return metrics_df['recall'].mean(), metrics_df['precision'].mean(), metrics_df['ndcg'].mean(), metrics_df['map'].mean()

def eval_metrics(col_name, train_df, test_df, n_users, n_items, user_Embed, item_Embed, batch_size, K):
  R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
  R[train_df[col_name['col_user']], train_df[col_name['col_item']]] = 1.0
  R = convert_to_sparse_tensor(R).coalesce()

  topk_recalls, topk_precisions, topk_ndcgs, topk_maps = [], [], [], []
  test_interaction = test_df.groupby(col_name["col_user"])[col_name["col_item"]].apply(list).reset_index()

  for start in tqdm(range(0, n_users, batch_size),desc ="Get relevance score"):
    end = min(start + batch_size, n_users)
    user_batch = user_Embed[start:end]  # Shape: (batch_size, embedding_dim)
    batch_relevance = torch.matmul(user_batch, item_Embed.T)

    # Compute relevance score for the current batch
    batch_relevance = torch.matmul(user_batch, item_Embed.T) # Shape: (batch_size, n_items)

    # Assuming R_tensor is a sparse tensor (Sparse COO format)
    batch_mask = torch.zeros((end - start, n_items), dtype=torch.float32)

    # Extract indices and values for the relevant batch
    indices = R.indices()
    values = R.values()

    batch_indices = (indices[0] >= start) & (indices[0] < end)
    filtered_indices = indices[:, batch_indices]
    filtered_indices[0] -= start
    filtered_values = values[batch_indices] * (-np.inf)
    batch_mask.index_put_(tuple(filtered_indices), filtered_values)

    batch_relevance = torch.matmul(user_batch, item_Embed.T) # Shape: (batch_size, n_items)
    batch_relevance += batch_mask
    batch_relevance = torch.nan_to_num(batch_relevance, nan=0.0)
    
    _, indices = torch.sort(batch_relevance, descending=True)
    topk_indicies = indices[:, :K]

    _recall, _precision, _ndcg, _map = get_metrics_with_chunks(col_name, topk_indicies, start, test_interaction, K)

    topk_recalls.append(_recall)
    topk_precisions.append(_precision)
    topk_ndcgs.append(_ndcg)
    topk_maps.append(_map)
  
  return np.mean(topk_recalls), np.mean(topk_precisions), np.mean(topk_ndcgs), np.mean(topk_maps)

def get_metrics_df_with_chunks(col_name, topk_relevance_indices, start, test_df, K):
  top_k_relevance_indicies_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
  top_k_relevance_indicies_df[col_name['col_user']] = top_k_relevance_indicies_df.index + start

  top_k_relevance_indicies_df['top_relevent_item'] = top_k_relevance_indicies_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
  top_k_relevance_indicies_df = top_k_relevance_indicies_df[[col_name['col_user'],'top_relevent_item']]

  metrics_df = top_k_relevance_indicies_df[top_k_relevance_indicies_df[col_name['col_user']].isin(np.unique(test_df[col_name['col_user']]))]

  return metrics_df

def recommend_k_items(col_name, train_df, test_df, n_users, n_items, user_Embed, item_Embed, batch_size, K):
  R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
  R[train_df[col_name['col_user']], train_df[col_name['col_item']]] = 1.0
  R = convert_to_sparse_tensor(R).coalesce()

  topk_recalls, topk_precisions, topk_ndcgs, topk_maps = [], [], [], []

  recommended_chunk_li = []

  for start in tqdm(range(0, n_users, batch_size),desc ="Get relevance score"):
    end = min(start + batch_size, n_users)
    user_batch = user_Embed[start:end]  # Shape: (batch_size, embedding_dim)
    batch_relevance = torch.matmul(user_batch, item_Embed.T)

    # Compute relevance score for the current batch
    batch_relevance = torch.matmul(user_batch, item_Embed.T) # Shape: (batch_size, n_items)

    # Assuming R_tensor is a sparse tensor (Sparse COO format)
    batch_mask = torch.zeros((end - start, n_items), dtype=torch.float32)

    # Extract indices and values for the relevant batch
    indices = R.indices()
    values = R.values()

    batch_indices = (indices[0] >= start) & (indices[0] < end)
    filtered_indices = indices[:, batch_indices]
    filtered_indices[0] -= start
    filtered_values = values[batch_indices] * (-np.inf)
    batch_mask.index_put_(tuple(filtered_indices), filtered_values)

    batch_relevance = torch.matmul(user_batch, item_Embed.T) # Shape: (batch_size, n_items)
    batch_relevance += batch_mask
    batch_relevance = torch.nan_to_num(batch_relevance, nan=0.0)
    
    _, indices = torch.sort(batch_relevance, descending=True)
    topk_indicies = indices[:, :K]

    recommended_chunk_li.append(get_metrics_df_with_chunks(col_name, topk_indicies, start, test_df, K))

  return pd.concat(recommended_chunk_li).reset_index(drop=True)
