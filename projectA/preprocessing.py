import numpy as np
import pandas as pd
import re
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize


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

def _clean_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Perform basic cleaning and feature scaling on numeric columns.

    The function imputes missing values using the median derived from the
    training set and then standardizes each numeric column (mean 0, variance
    1) using ``StandardScaler``. The same transformers are applied to the test
    set to avoid data leakage.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training features.
    test_df : pandas.DataFrame
        Test features (same columns as ``train_df``).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Cleaned & scaled versions of ``train_df`` and ``test_df``.
    """
    # identify numeric columns only
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) == 0:
        # nothing to do, return original
        return train_df, test_df

    # impute missing values with median (fit on train only)
    imputer = SimpleImputer(strategy='median')
    train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])

    # standardize features (fit on train only)
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    return train_df, test_df


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
    x_BERT = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
    x_BERT = normalize(x_BERT, norm='l2', axis=1)
    return x_BERT

def preprocess():
    x_train, y_train, x_test = read_data()

    # drop columns that we won't use for modelling
    # drop_cols = ['text', 'author', 'title', 'passage_id']

    drop_cols = ['text', 'author', 'title', 'passage_id', 'sentence_count', 'char_count',
                 'word_count', 'info_wordtypes',
                 'punctuation_frequency', 'function_words_count',
                 'pronoun_freq', 'sentiment_polarity', 'sentiment_subjectivity',
                 'info_words', 'info_syllables', 'info_type_token_ratio',
                 'info_words_per_sentence', 'info_characters',
                 'readability_SMOGIndex','readability_RIX', 'readability_DaleChallIndex',
                  'readability_GunningFogIndex', 'readability_LIX','readability_Kincaid', 'readability_ARI'
                 ]
    x_train.drop(columns=drop_cols, inplace=True)
    x_test.drop(columns=drop_cols, inplace=True)

    # clean and scale numerical features
    x_train, x_test = _clean_and_scale(x_train, x_test)

    # append BERT features to each dataframe
    x_tr_bert = create_BERT(x_train, 'x_train_BERT_embeddings.npz')
    x_ts_bert = create_BERT(x_test, 'x_test_BERT_embeddings.npz')


    return x_tr_bert, y_train, x_ts_bert
    
if __name__ == '__main__':
    x_tr_bert, y_train, x_ts_bert = preprocess()
    print("x_ts_bert columns:", x_ts_bert.columns.tolist())
    print("x_tr_bert shape:", x_tr_bert.shape)
    print("x_tr", x_tr_bert.head())
