import pandas as pd
import numpy as np
import pandera.pandas as pa
import json
import os
import datetime

try:
    from evidently.report import Report
except ImportError:
    try:
        from evidently.legacy.report import Report
    except ImportError:
        try:
            from evidently import Report
        except ImportError:
            Report = None

# Dynamic discovery of presets to handle version differences
DataDriftPreset = TargetDriftPreset = DataQualityPreset = RegressionPreset = None

def _discover_presets():
    global DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
    import importlib
    
    # Try direct imports first (most robust if we know the path)
    presets_map = {
        'DataDriftPreset': [
            'evidently.legacy.metric_preset',
            'evidently.metric_preset.data_drift',
            'evidently.metric_preset',
            'evidently.metric_presets',
            'evidently.metrics.preset',
            'evidently.presets'
        ],
        'RegressionPreset': [
            'evidently.legacy.metric_preset',
            'evidently.metric_preset.regression',
            'evidently.metric_preset',
            'evidently.metric_presets',
            'evidently.metrics.preset',
            'evidently.presets',
            'evidently.presets.regression'
        ],
        'DataQualityPreset': [
            'evidently.legacy.metric_preset',
            'evidently.metric_preset.data_quality',
            'evidently.metric_preset',
            'evidently.metric_presets',
            'evidently.metrics.preset'
        ],
        'TargetDriftPreset': [
            'evidently.legacy.metric_preset',
            'evidently.metric_preset.target_drift',
            'evidently.metric_preset',
            'evidently.metric_presets',
            'evidently.metrics.preset'
        ]
    }
    
    for var_name, modules in presets_map.items():
        found_val = None
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, var_name):
                    found_val = getattr(mod, var_name)
                    break
            except ImportError:
                continue
        
        if var_name == 'DataDriftPreset': DataDriftPreset = found_val
        elif var_name == 'RegressionPreset': RegressionPreset = found_val
        elif var_name == 'DataQualityPreset': DataQualityPreset = found_val
        elif var_name == 'TargetDriftPreset': TargetDriftPreset = found_val

_discover_presets()

# Re-alias for convenience if found
if pa:
    Column = pa.Column
    Check = pa.Check
    DataFrameSchema = pa.DataFrameSchema

# 1. PANDERA SCHEMAS
# Define expectations for our data to catch errors early

ebay_clean_schema = DataFrameSchema({
    "Title": Column(str, Check.str_length(min_value=3)),
    "Price": Column(str),
    "price_cleaned": Column(float, Check.greater_than(0)),
    "average_rating": Column(float, Check.in_range(0, 5)),
    "num_reviews": Column(float, Check.greater_than_or_equal_to(0)),
    "original_price": Column(float, Check.greater_than_or_equal_to(0)),
    "match_confidence": Column(float, Check.in_range(0, 1)),
})

feature_schema = DataFrameSchema({
    "current_price": Column(float, Check.greater_than(0), required=False),
    "log_reviews": Column(float, Check.greater_than_or_equal_to(0)),
    "is_apple": Column(int, Check.isin([0, 1])),
    "has_brand": Column(int, Check.isin([0, 1])),
})

# 2. LOGGING HELPERS

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def log_step_results(step_name, metrics):
    """Save step metrics to a JSON log file"""
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f"{step_name}.json")
    
    # Add timestamp
    metrics["timestamp"] = datetime.datetime.now().isoformat()
    
    # Read existing or create new
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    
    if not isinstance(history, list):
        history = [history]
        
    history.append(metrics)
    
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4, cls=NpEncoder)
    
    print(f"📝 Logged results for {step_name} to {log_path}")

# 3. EVIDENTLY AI REPORTS

def generate_drift_report(reference_data, current_data, output_name="data_drift"):
    """Generate an Evidently AI drift report"""
    os.makedirs("data/reports", exist_ok=True)
    
    if Report is None:
        print("⚠️ Skipping drift report: Evidently not installed correctly.")
        return None
        
    try:
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ])
        
        report.run(reference_data=reference_data, current_data=current_data)
        
        report_path = f"data/reports/{output_name}.html"
        report.save_html(report_path)
        print(f"📊 Generated drift report: {report_path}")
        return report_path
    except Exception as e:
        print(f"⚠️ Failed to generate drift report: {e}")
        return None

def generate_model_performance_report(y_true, y_pred, output_name="model_performance"):
    """Generate an Evidently AI regression performance report"""
    os.makedirs("data/reports", exist_ok=True)
    
    if Report is None:
        print("⚠️ Skipping performance report: Evidently not installed correctly.")
        return None
        
    # Create DataFrames for Evidently
    # Convert to standard types to avoid issues with numpy versions
    cur_df = pd.DataFrame({
        "target": np.array(y_true).flatten(), 
        "prediction": np.array(y_pred).flatten()
    })
    
    try:
        report = Report(metrics=[
            RegressionPreset(),
        ])
        
        # In newer versions or legacy, ref_data=None might be tricky,
        # but evidently usually supports single dataset for performance
        report.run(reference_data=None, current_data=cur_df)
        
        report_path = f"data/reports/{output_name}.html"
        report.save_html(report_path)
        print(f"📊 Generated model performance report: {report_path}")
        return report_path
    except Exception as e:
        print(f"⚠️ Failed to generate performance report: {e}")
        print("   This is often due to compatibility issues with newer Pandas/Numpy versions.")
        return None

def validate_dataframe(df, schema, step_name):
    """Validate a dataframe using Pandera and log result"""
    try:
        schema.validate(df, lazy=True)
        print(f"✅ {step_name}: Validation passed.")
        log_step_results(f"{step_name}_validation", {"status": "success", "rows": len(df)})
        return True
    except pa.errors.SchemaErrors as err:
        print(f"❌ {step_name}: Validation FAILED!")
        # Log failure with details
        log_step_results(f"{step_name}_validation", {
            "status": "failure", 
            "errors": str(err.failure_cases.head().to_dict()),
            "rows": len(df)
        })
        # For now, we print and continue, but could raise exception to halt pipeline
        print(err.failure_cases.head())
        return False
