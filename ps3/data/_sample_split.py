import hashlib
import pandas as pd
import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    columns = id_column if isinstance(id_column, list) else [id_column]
        
    def int_from_row(r):
        vals = [str(v) if pd.notna(v) else "__MISSING__" for v in r]
        key_str = "|".join(vals)
        b = key_str.encode("utf-8")
        h = hashlib.md5(b).digest()
        return int.from_bytes(h, byteorder='big')
    
    threshold = int(training_frac * 100)
    ids = df[columns].apply(int_from_row, axis=1)
    buckets = ids % 100
    df['sample'] = np.where(buckets < threshold, 'train', 'test')
    return df
