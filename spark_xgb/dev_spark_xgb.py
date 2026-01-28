import random
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    Imputer,
    VectorAssembler,
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark2pmml import PMMLBuilder
from xgboost.spark import SparkXGBClassifier
import os

# Avoid potential issues on macOS with forking
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

OUTPUT_DIR = "out"
PMML_PATH = f"{OUTPUT_DIR}/full_pipeline_spark_xgb.pmml"

# Train Data Schema
LABEL_COL = "label"
CAT_COLS = ["merchant", "channel"]      # categorical/string columns
NUM_COLS = ["amount", "age", "days"]    # numeric columns


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def create_train_data(spark: SparkSession, n: int = 3000, seed: int = 42):
    random.seed(seed)
    merchants = ["A-Mart", "B-Store", "C-Shop", "D-Market"]
    channels = ["online", "pos", "mobile"]
    rows = []
    for _ in range(n):
        merchant = random.choice(merchants)
        channel = random.choice(channels)
        amount = max(0.0, random.gauss(80, 50))
        age = random.uniform(18, 80)
        days = random.uniform(0, 365)
        # generate a score based on features and add random noise
        score = 0.02 * amount + (1.2 if channel == "online" else 0.2) + (0.5 if merchant in ["C-Shop", "D-Market"] else 0.0)
        score += random.gauss(0, 0.5)
        label = 1 if score > 2.6 else 0
        # add some missing values for 'age'
        if random.random() < 0.02:
            age_val = None
        else:
            age_val = float(age)
        rows.append((int(label), merchant, channel, float(amount), age_val, float(days)))
    return spark.createDataFrame(rows, [LABEL_COL] + CAT_COLS + NUM_COLS)

def main():
    ensure_dir(OUTPUT_DIR)
        
    spark = (
    SparkSession.builder
        .appName("SparkML_FullPipeline_To_PMML")
        .config(
        "spark.jars.packages",
        ",".join([
            "org.jpmml:pmml-sparkml:2.2.6",
            "org.jpmml:pmml-sparkml-xgboost:2.2.6",
            "ml.dmlc:xgboost4j-spark_2.12:1.7.6",
            "ml.dmlc:xgboost4j_2.12:1.7.6",
        ])
        )
        .getOrCreate()
    )

    # Load actual data here 
    df = create_train_data(spark)
    
    df = (
        df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("double"))  # Changed to double for XGBoost
          .filter(F.col(LABEL_COL).isin(0, 1))
    )
    print("Label distribution:")
    df.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

    # Build preprocessing pipeline
    # Convert string/categorical to integer
    cat_indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CAT_COLS
    ]
    # Convert categorical column to one-hot encoding
    ohe = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in CAT_COLS],
        outputCols=[f"{c}_ohe" for c in CAT_COLS],
        handleInvalid="keep",
    )
    # Impute missing numeric values
    imputer = Imputer(
        inputCols=NUM_COLS,
        outputCols=[f"{c}_imp" for c in NUM_COLS],
        strategy="mean",  # Explicitly set strategy
    )
    # Assemble all features into a single vector column
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in CAT_COLS] + [f"{c}_imp" for c in NUM_COLS],
        outputCol="features",
        handleInvalid="keep",
    )

    # Build modeling pipeline with XGBoost
    clf = SparkXGBClassifier(
        label_col=LABEL_COL,
        features_col="features",
        raw_prediction_col="rawPrediction",
        max_depth=6,
        seed=42,
    )
    # Add all stages to a single pipeline
    pipeline = Pipeline(stages=cat_indexers + [ohe, imputer, assembler, clf])
        
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Hyperparameter tuning 
    param_grid = (
        ParamGridBuilder()
        .addGrid(clf.max_depth, [4, 6, 8])
        .build()
    )

    # Model evaluation metrics
    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    # Cross-validation
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4,
        seed=42,
    )

    print("Training model with cross-validation...")
    cv_model = cv.fit(train)
    
    # Choose best pipeline model from CV
    pipeline_model = cv_model.bestModel

    # Evaluate on test
    print("\nEvaluating on test set...")
    preds = pipeline_model.transform(test)
    preds.select(LABEL_COL, "probability", "prediction").show(10, truncate=False)
    auc = evaluator.evaluate(preds)
    print(f"Test AUC = {auc:.4f}")

    # Export the full pipeline (preprocessing + model) to PMML
    print(f"\nExporting model to PMML...")
    pmml_builder = PMMLBuilder(spark, train, pipeline_model)
    
    # Verify the schema before building
    pmml_builder.verify(train.sample(False, 0.01, seed=42).limit(10))
    
    # Build and save PMML file
    pmml_builder.buildFile(PMML_PATH)
    print(f"Successfully exported PMML (preprocessing + XGBoost model): {PMML_PATH}")
    
    spark.stop()


if __name__ == "__main__":
    main()