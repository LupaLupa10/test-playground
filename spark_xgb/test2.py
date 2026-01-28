import os
import sys
import random
from pathlib import Path
import urllib.request

# macOS ARM-specific configurations
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

BASE_DIR = Path(__file__).resolve().parent
JARS_DIR = BASE_DIR / "jars"
NATIVE_LIB_DIR = BASE_DIR / "native_libs"

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def download_jar(url: str, filename: str) -> str:
    """Download JAR file if not exists"""
    ensure_dir(str(JARS_DIR))
    jar_path = str(JARS_DIR / filename)
    
    if Path(jar_path).exists():
        print(f"✓ {filename}")
        return jar_path
    
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, jar_path)
        print(f"✓ Downloaded {filename}")
        return jar_path
    except Exception as e:
        print(f"✗ Failed: {e}")
        return ""


def spark_packages():
    """
    Return Maven coordinates for Spark package resolution (transitive deps included).

    This avoids manual jar curation and, critically, ensures JPMML-SparkML matches Spark 3.4.X.
    Compatibility matrix (pyspark2pmml README):
      Spark 3.4.X -> JPMML-SparkML 3.0.X (latest 3.0.9)
    """
    JPMML_SPARKML_VERSION = "3.0.9"
    XGBOOST4J_SPARK_VERSION = "2.0.3"

    return [
        f"org.jpmml:pmml-sparkml:{JPMML_SPARKML_VERSION}",
        f"org.jpmml:pmml-sparkml-xgboost:{JPMML_SPARKML_VERSION}",
        f"ml.dmlc:xgboost4j-spark_2.12:{XGBOOST4J_SPARK_VERSION}",
        f"ml.dmlc:xgboost4j_2.12:{XGBOOST4J_SPARK_VERSION}",
    ]


def setup_native_library():
    """Setup XGBoost native library"""
    ensure_dir(str(NATIVE_LIB_DIR))
    lib_path = str(NATIVE_LIB_DIR / "libxgboost.dylib")
    
    if Path(lib_path).exists():
        return lib_path
    
    print("Setting up XGBoost native library...")
    
    try:
        import xgboost
        xgb_base = Path(xgboost.__file__).parent
        for file in xgb_base.rglob("libxgboost.dylib"):
            print(f"✓ Found library: {file}")
            import shutil
            shutil.copy(str(file), lib_path)
            return lib_path
    except:
        pass
    
    # Try Homebrew
    homebrew_lib = "/opt/homebrew/lib/libxgboost.dylib"
    if Path(homebrew_lib).exists():
        import shutil
        shutil.copy(homebrew_lib, lib_path)
        return lib_path
    
    return None


# Setup Spark packages and native library BEFORE importing PySpark
packages = spark_packages()
native_lib = setup_native_library()

if not native_lib:
    print("ERROR: XGBoost library not found. Run: pip install xgboost")
    sys.exit(1)

lib_dir = str(Path(native_lib).parent.resolve())
os.environ["DYLD_LIBRARY_PATH"] = f"{lib_dir}:/opt/homebrew/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")

# Set PYSPARK environment variables BEFORE importing
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_SUBMIT_ARGS"] = f\"--packages {','.join(packages)} pyspark-shell\"

# NOW import PySpark
try:
    from pyspark.sql import SparkSession, functions as F
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.wrapper import JavaEstimator, JavaModel
    from pyspark.ml.param.shared import HasLabelCol, HasFeaturesCol
except ModuleNotFoundError as e:
    raise RuntimeError(
        "PySpark is not installed in this Python environment.\n"
        "Install the pinned dependencies from spark_xgb/requirements.txt, e.g.:\n"
        "  pip install -r spark_xgb/requirements.txt\n"
    ) from e

try:
    from pyspark2pmml import PMMLBuilder
except ModuleNotFoundError as e:
    raise RuntimeError(
        "pyspark2pmml is not installed in this Python environment.\n"
        "Install the pinned dependencies from spark_xgb/requirements.txt, e.g.:\n"
        "  pip install -r spark_xgb/requirements.txt\n"
    ) from e


OUTPUT_DIR = "out"
PMML_PATH = f"{OUTPUT_DIR}/full_pipeline_xgboost4j.pmml"
LABEL_COL = "label"
CAT_COLS = ["merchant", "channel"]
NUM_COLS = ["amount", "age", "days"]


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

        score = 0.02 * amount + (1.2 if channel == "online" else 0.2) + (
            0.5 if merchant in ["C-Shop", "D-Market"] else 0.0
        )
        score += random.gauss(0, 0.5)
        label = 1.0 if score > 2.6 else 0.0

        age_val = None if random.random() < 0.02 else float(age)
        rows.append((label, merchant, channel, float(amount), age_val, float(days)))

    return spark.createDataFrame(rows, [LABEL_COL] + CAT_COLS + NUM_COLS)


class XGBoost4JClassifier(JavaEstimator, HasLabelCol, HasFeaturesCol):
    _java_class = "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier"

    def __init__(self):
        super().__init__()
        self._java_obj = self._new_java_obj(self._java_class, self.uid)

    def _create_model(self, java_model):
        return JavaModel(java_model)

    def setLabelCol(self, value: str):
        self._java_obj.setLabelCol(value)
        return self

    def setFeaturesCol(self, value: str):
        self._java_obj.setFeaturesCol(value)
        return self

    def setPredictionCol(self, value: str):
        self._java_obj.setPredictionCol(value)
        return self

    def setProbabilityCol(self, value: str):
        self._java_obj.setProbabilityCol(value)
        return self

    def setRawPredictionCol(self, value: str):
        self._java_obj.setRawPredictionCol(value)
        return self

    def setObjective(self, value: str):
        self._java_obj.setObjective(value)
        return self

    def setNumRound(self, value: int):
        self._java_obj.setNumRound(int(value))
        return self

    def setNumWorkers(self, value: int):
        self._java_obj.setNumWorkers(int(value))
        return self

    def setMaxDepth(self, value: int):
        self._java_obj.setMaxDepth(int(value))
        return self

    def setEta(self, value: float):
        self._java_obj.setEta(float(value))
        return self

    def setSubsample(self, value: float):
        self._java_obj.setSubsample(float(value))
        return self

    def setColsampleBytree(self, value: float):
        self._java_obj.setColsampleBytree(float(value))
        return self

    def setMissing(self, value: float):
        self._java_obj.setMissing(float(value))
        return self
    
    def setTreeMethod(self, value: str):
        self._java_obj.setTreeMethod(value)
        return self


def main():
    # Make output location stable regardless of current working directory
    output_dir = BASE_DIR / OUTPUT_DIR
    ensure_dir(str(output_dir))
    pmml_path = str(output_dir / Path(PMML_PATH).name)

    if not packages:
        raise RuntimeError("No Spark package coordinates configured.")

    print("="*60)
    print("Starting Spark...")
    print("="*60)

    spark = (
        SparkSession.builder
        .appName("XGBoost4J PMML")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.extraLibraryPath", lib_dir)
        .config("spark.executor.extraLibraryPath", lib_dir)
        .config("spark.jars.packages", ",".join(packages))
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")

    print("\nCreating data...")
    df = create_train_data(spark).withColumn(LABEL_COL, F.col(LABEL_COL).cast("double"))
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    print("Building pipeline...")
    
    cat_indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CAT_COLS
    ]

    ohe = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in CAT_COLS],
        outputCols=[f"{c}_ohe" for c in CAT_COLS],
        handleInvalid="keep",
        dropLast=False,
    )

    imputer = Imputer(
        inputCols=NUM_COLS,
        outputCols=[f"{c}_imp" for c in NUM_COLS],
        strategy="mean",
    )

    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in CAT_COLS] + [f"{c}_imp" for c in NUM_COLS],
        outputCol="features",
        handleInvalid="keep",
    )

    clf = (
        XGBoost4JClassifier()
        .setLabelCol(LABEL_COL)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setProbabilityCol("probability")
        .setRawPredictionCol("rawPrediction")
        .setObjective("binary:logistic")
        .setNumRound(100)
        .setNumWorkers(2)
        .setMaxDepth(6)
        .setEta(0.1)
        .setSubsample(0.8)
        .setColsampleBytree(0.8)
        .setMissing(0.0)
        .setTreeMethod("hist")
    )

    pipeline = Pipeline(stages=cat_indexers + [ohe, imputer, assembler, clf])

    print("\nTraining...")
    pipeline_model = pipeline.fit(train)
    print("✓ Training done")

    print("\nEvaluating...")
    preds = pipeline_model.transform(test)
    
    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator.evaluate(preds)
    print(f"Test AUC = {auc:.4f}")

    print("\nExporting to PMML...")
    PMMLBuilder(spark, train, pipeline_model).buildFile(pmml_path)
    
    file_size = Path(pmml_path).stat().st_size / 1024
    print(f"✅ Success! {pmml_path} ({file_size:.2f} KB)")

    spark.stop()


if __name__ == "__main__":
    main()