import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(filepath, low_memory=False)

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary labels: 1 if Tc > 10K, else 0."""
    df['label'] = (df['tc'] > 10).astype(int)
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only numerical features, dropping unnecessary columns."""
    drop_columns = [
        'SC_formula', 'SC_class', 'MP_parent', 'MP_id', 'chemical_composition',
        'synth_doped', 'magnetic_type', 'exchange_symmetry'
    ]
    df = df.drop(columns=drop_columns, errors='ignore')

    # Drop SOAP descriptors if needed
    df = df[[col for col in df.columns if not col.startswith('SOAP_')]]

    return df

def preprocess(filepath: str, output_dir: str):
    """Full preprocessing pipeline."""
    df = load_data(filepath)
    df = create_labels(df)
    df = select_features(df)

    df = df.dropna()

    features = df.drop(columns=['tc', 'label'])
    labels = df['label']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((features_scaled, labels.values), os.path.join(output_dir, 'processed.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

if __name__ == "__main__":
    preprocess("data/3DSC_database_cleaned.csv", "processed")