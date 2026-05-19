# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split

class MarketingDataPipeline:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_and_clean(self) -> pd.DataFrame:
        """Loads dataset and prepares treatment, outcome, and confounders."""
        # Simulating Kaggle Causal Digital Marketing structural expectations
        self.df = pd.read_csv(self.filepath)
        
        # Binary treatment: 1 if exposed to ad campaign, 0 if control group
        self.df['treatment'] = self.df['ad_exposed'].astype(int)
        
        # Target variable: Continuous revenue generated or conversion value
        self.df['outcome'] = self.df['revenue_outcomes'].astype(float)
        
        return self.df

    def split_features(self, df: pd.DataFrame):
        """Splits data into Confounders (X), Treatment (T), and Outcome (Y)."""
        # Select background traits that influence both ad exposure and revenue
        confounder_cols = [
            'past_spend', 'impressions', 'clicks', 
            'channel_Search', 'channel_Social', 'channel_Email'
        ]
        
        # Dynamically handle missing columns if dummy variables weren't created yet
        available_confounders = [col for col in confounder_cols if col in df.columns]
        
        X = df[available_confounders].fillna(0).values
        W = X.copy() # Controls for standard nuisance models
        T = df['treatment'].values
        Y = df['outcome'].values
        
        return train_test_split(X, W, T, Y, test_size=0.2, random_state=42)
