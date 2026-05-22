# main.py
import os
import pandas as pd
import numpy as np
import pandas as pd
from src.data_processing import MarketingDataPipeline
from src.causal_models import CausalUpliftEngine 

def run_pipeline():
    # Define production directories
    data_path = os.path.join("data", "causal_digital_marketing_campaign_dataset.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Please download the Causal Digital Marketing Campaign dataset "
            f"from Kaggle and place it in: {data_path}"
        )
        
    # 1. Processing pipeline execution
    pipeline = MarketingDataPipeline(data_path)
    cleaned_df = pipeline.load_and_clean()
    
    # 2. Partition data
    X_tr, X_te, W_tr, W_te, T_tr, T_te, Y_tr, Y_te = pipeline.split_features(cleaned_df)
    
    # 3. Model Engine execution
    engine = CausalUpliftEngine()
    engine.fit_pipeline(X_tr, W_tr, T_tr, Y_tr)
    
    # 4. Generate Predictions 
    predicted_uplifts = engine.estimate_effects(X_te)
    
    # 5. Output Summary Diagnostics
    print("\n" + "="*40)
    print("CAUSAL INFERENCE ESTIMATION COMPLETE")
    print("="*40)
    print(f"Mean Predicted Campaign Uplift per user: ${np.mean(predicted_uplifts):.2f}")
    print(f"Max Campaign ROI Opportunity found:     ${np.max(predicted_uplifts):.2f}")
    print(f"Percentage of negative responders:      {np.mean(predicted_uplifts < 0)*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
