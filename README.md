
### Spark_Mllib 
**Approach:** Using native-spark for data processing + machine learning library. Able to have full pipeline (data processing and model) and export to PMML. However no XGBoost and AutoML. 

**Script:** dev_spark_mllib.py (dev), inference_spark_mllib.py (inference).

Apache Spark: https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier

Run with jars on classpath (example using --packages):
```bash
spark-submit \
  --packages org.jpmml:pmml-sparkml:3.3.1,org.jpmml:jpmml-sparkml:3.3.1 \
  dev_spark_mllib.py
```

Or run using python: 
```bash
python dev_spark_mllib.py on the terminal 
```



<!-- jpmml-h20: convert H20.ai models to PMML

JPMML-SparkML: core library and convert Apache Spark ML pipelines to PMML

PySpark2PMML: wrapper for JPMML-SparkML

xgboost4j: JVM package 

JPMML-SparkML-XGBoost  -->
