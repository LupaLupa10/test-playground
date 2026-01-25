import pandas as pd
from pypmml import Model

PMML_PATH = "out/full_pipeline_spark_mllib.pmml"


def create_test_data() -> pd.DataFrame:
    rows = [
        {"merchant": "A-Mart",   "channel": "online", "amount": 210.0, "age": 36.0,  "days": 12.0},
        {"merchant": "B-Store",  "channel": "pos",    "amount": 15.0,  "age": None,  "days": 200.0},
        {"merchant": "D-Market", "channel": "mobile", "amount": 120.0, "age": 52.0,  "days": 5.0},
        {"merchant": "C-Shop",   "channel": "online", "amount": 80.0,  "age": 24.0,  "days": 320.0},
    ]
    return pd.DataFrame(rows)


def main():
    model = Model.load(PMML_PATH)
    print(f"Loaded PMML: {PMML_PATH}")

    X = create_test_data()
    print("\nInput rows:")
    print(X)

    # predict using PMML pipeline, which includes preprocessing + model
    yhat = model.predict(X)
    print("\nPMML output:")
    print(yhat)

    pred_cols = [c for c in yhat.columns if "predicted" in c.lower() or c.lower() in ("prediction", "predictedlabel")]
    if pred_cols:
        print(f"\nPredicted label column (detected): {pred_cols[0]}")


if __name__ == "__main__":
    main()
