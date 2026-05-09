import mlflow
import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon

EXPERIMENT_NAME = "Home_Credit_Default_Risk"
MLFLOW_DB_URI = "sqlite:///mlflow.db"
REPORT_DIR = "src/evaluation/report/"
REPORT_PATH = os.path.join(REPORT_DIR, "model_comparison_report.md")

class MLflowEvaluator:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_DB_URI)
        self.client = mlflow.tracking.MlflowClient()
        try:
            self.experiment_id = self.client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        except AttributeError:
            raise ValueError(f"Experiment '{EXPERIMENT_NAME}' wasn't found {MLFLOW_DB_URI}")
        
    def get_run_data(self, parent_name, version_type):
        """
        Retrieves data while accounting for the nesting structure:
        - v1 and v2: look for the child ‘Final_Production’
        - v3: retrieve metrics from the parent itself
        """
        # Find parent's runs by name
        parent_runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.mlflow.runName = '{parent_name}'"
        )
        
        if parent_runs.empty:
            return None

        target_runs_list = []

        for _, parent in parent_runs.iterrows():
            if version_type in ['v1', 'v2']:
                # Find nested run "Final_Production" for this parent
                child_runs = mlflow.search_runs(
                    experiment_ids=[self.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{parent.run_id}' AND tags.mlflow.runName = 'Final_Production'"
                )
                if not child_runs.empty:
                    target_runs_list.append(child_runs.iloc[0])
            else:
                # Take the parent run for v3
                target_runs_list.append(parent)

        if not target_runs_list:
            return None

        # Transform list to DataFrame to make sorting easier
        df_candidates = pd.DataFrame(target_runs_list)
        
        # Check whether column exist or not
        metric_col = 'metrics.cv_mean_roc_auc'
        if metric_col not in df_candidates.columns:
            return None

        # Sort and get the best model if there are multiple models
        best_run = df_candidates.sort_values(metric_col, ascending=False).iloc[0]
        
        # Remove'metrics.' prefix
        data_dict = best_run.to_dict()
        clean_metrics = {k.replace('metrics.', ''): v for k, v in data_dict.items() if str(k).startswith('metrics.')}
        
        # Add the model info
        clean_metrics['model_family'] = parent_name.replace('_non_hpo', '').replace('_base', '')
        clean_metrics['version'] = version_type
        return clean_metrics

    def collect_all_data(self):
        # Run's mapping
        mapping = [
            ('logreg', 'v1'), ('xgb', 'v1'), ('catboost', 'v1'), ('lgbm', 'v1'),
            ('logreg_non_hpo', 'v2'), ('xgb_non_hpo', 'v2'), ('catboost_non_hpo', 'v2'), ('lgbm_non_hpo', 'v2'),
            ('logreg_base', 'v3'), ('xgb_base', 'v3'), ('catboost_base', 'v3'), ('lgbm_base', 'v3')
        ]
        
        results = []
        for p_name, v_type in mapping:
            print(f"Data extraction for: {p_name} ({v_type})...")
            run_data = self.get_run_data(p_name, v_type)
            if run_data:
                results.append(run_data)
        
        return pd.DataFrame(results)

def generate_report(df):
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report: Home Credit Default Risk\n\n")
        
        # Summary Table
        f.write("## Summary results (ROC AUC)\n")
        version_map = {'v1': 'Full + Optuna', 'v2': 'Full (No HPO)', 'v3': 'Base (No HPO)'}
        display_df = df.copy()
        display_df['version'] = display_df['version'].map(version_map)
        
        pivot_auc = display_df.pivot(index='model_family', columns='version', values='cv_mean_roc_auc')
        # Columns sorting
        cols = [v for v in version_map.values() if v in pivot_auc.columns]
        f.write(pivot_auc[cols].to_markdown() + "\n\n")

        # Hypothesis testing
        f.write("## Hypothesis testing\n\n")

        # First Hypothesis
        try:
            lgbm_v2 = df[(df.model_family == 'lgbm') & (df.version == 'v2')].iloc[0]
            lgbm_v3 = df[(df.model_family == 'lgbm') & (df.version == 'v3')].iloc[0]
            diff_h1 = lgbm_v2.cv_mean_roc_auc - lgbm_v3.cv_mean_roc_auc
            h1_status = "Confirmed!" if diff_h1 >= 0.015 else "Not confirmed"
            
            f.write(f"### Hypothesis 1: The Value of External Tables (Full vs Base)\n")
            f.write(f"- **Full Data AUC**: {lgbm_v2.cv_mean_roc_auc:.4f}, **Base Table AUC**: {lgbm_v3.cv_mean_roc_auc:.4f}\n")
            f.write(f"- **Increase**: {diff_h1:.4f} (needed >= 0.015)\n")
            f.write(f"- **Result**: {h1_status}\n\n")
        except Exception as e:
            f.write(f"### Hypothesis 1: Calculation error\n\n")

        # Second Hypothesis
        try:
            hpo_gains = []
            for model in ['lgbm', 'xgb', 'catboost']:
                v1_auc = df[(df.model_family == model) & (df.version == 'v1')]['cv_mean_roc_auc'].values[0]
                v2_auc = df[(df.model_family == model) & (df.version == 'v2')]['cv_mean_roc_auc'].values[0]
                hpo_gains.append(v1_auc - v2_auc)
            
            avg_gain = np.mean(hpo_gains)
            h2_status = "Confirmed!" if avg_gain >= 0.005 else "Not confirmed"
            
            f.write(f"### Hypothesis 2: The effectiveness of Optuna (v1 vs v2)\n")
            f.write(f"- **Average increase in AUC**: {avg_gain:.5f} (needed >= 0.005)\n")
            f.write(f"- **Result**: {h2_status}\n\n")
        except Exception as e:
            f.write(f"### Hypothesis 2: Calculation error\n\n")

if __name__ == "__main__":
    evaluator = MLflowEvaluator()
    data = evaluator.collect_all_data()
    
    if not data.empty:
        generate_report(data)
        print(f"\nReport has been created: {REPORT_PATH}")
    else:
        print("\nError! Unable to collect data. Check the run names in MLflow")