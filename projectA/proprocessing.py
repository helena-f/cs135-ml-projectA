import numpy as np
import pandas as pd
import re
import os


def read_data():
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    return x_train_df, y_train_df, x_test_df

def load_arr_from_npz(npz_path):
    ''' Load array from npz compressed file given path

    Returns
    -------
    arr : numpy ndarray
    '''
    npz_file_obj = np.load(npz_path)
    arr = npz_file_obj.f.arr_0.copy() # Rely on default name from np.savez
    npz_file_obj.close()
    return arr

def create_BERT(df, npz_filename):
    """Return a new DataFrame containing the original columns plus BERT features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe whose rows correspond to the embeddings file.
    npz_filename : str
        Name of the npz file located in the ``data_readinglevel`` directory.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with additional columns ``bert_0`` ... ``bert_{H-1}``.
    """
    data_dir = 'data_readinglevel'
    emb_path = os.path.join(data_dir, npz_filename)

    xBERT_NH = load_arr_from_npz(emb_path)
    assert xBERT_NH.ndim == 2, "expected 2‑D array of embeddings"
    if xBERT_NH.shape[0] != df.shape[0]:
        raise ValueError(f"number of rows in embeddings ({xBERT_NH.shape[0]})"
                         f" does not match dataframe ({df.shape[0]})")

    # create column names for each embedding dimension
    H = xBERT_NH.shape[1]
    bert_cols = [f'bert_{i}' for i in range(H)]
    bert_df = pd.DataFrame(xBERT_NH, columns=bert_cols, index=df.index)

    # concatenate and return
    return pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

def preprocess():
    x_train, y_train, x_test = read_data()

    # drop columns
    x_train.drop( columns=['text', 'author', 'title', 'passage_id'], inplace=True)
    x_test.drop(columns=['text', 'author', 'title', 'passage_id'], inplace=True)

    # append BERT features to each dataframe
    x_tr_bert = create_BERT(x_train, 'x_train_BERT_embeddings.npz')
    x_ts_bert = create_BERT(x_test, 'x_test_BERT_embeddings.npz')

    return x_tr_bert, y_train, x_ts_bert
    
if __name__ == '__main__':
    x_tr_bert, y_train, x_ts_bert = preprocess()
    print("x_ts_bert columns:", x_ts_bert.columns.tolist())
    print("x_tr_bert shape:", x_tr_bert.shape)
    print("x_tr", x_tr_bert.head())
