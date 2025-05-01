from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine Quality Training") \
    .getOrCreate()

# === Load training dataset ===
training_data = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .option("quote", "\"") \
    .load("/app/TrainingDataset.csv")

# === Load validation dataset ===
validation_data = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .option("quote", "\"") \
    .load("/app/ValidationDataset.csv")

# === Clean column names ===
def clean_column(col_name):
    return col_name.replace('"', '').replace("'", "").strip()

training_data = training_data.toDF(*[clean_column(c) for c in training_data.columns])
validation_data = validation_data.toDF(*[clean_column(c) for c in validation_data.columns])

# === Extract feature columns ===
feature_columns = training_data.columns
print("Cleaned Columns:", feature_columns)

if "quality" in feature_columns:
    feature_columns.remove("quality")
else:
    raise Exception("Column 'quality' not found even after cleaning.")

# === Create label column and cache ===
training_data = training_data.withColumn("label", training_data["quality"].cast("integer"))
validation_data = validation_data.withColumn("label", validation_data["quality"].cast("integer"))

training_data = training_data.cache()
validation_data = validation_data.cache()

# === Vector Assembler ===
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# === Logistic Regression model ===
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# === Pipeline ===
pipeline = Pipeline(stages=[assembler, lr])

# === Parameter grid for tuning ===
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [10, 20, 50]) \
    .build()

# === Evaluator (F1 Score) ===
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

# === CrossValidator ===
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

# === Train model ===
cvModel = crossval.fit(training_data)

# === Get best model ===
best_model = cvModel.bestModel

# === Predict and evaluate on validation data ===
predictions = best_model.transform(validation_data)
f1_score = evaluator.evaluate(predictions)
print(f"\n✅ F1 Score on validation data: {f1_score}\n")

# === Save the best model ===
model_output_path = "/home/jovyan/wine_quality_model"
best_model.write().overwrite().save(model_output_path)
print(f"✅ Model saved to: {model_output_path}")

model_output_path = "/home/jovyan/wine_quality_model"
best_model.write().overwrite().save(model_output_path)
print(f"✅ Model saved to: {model_output_path}")


# === Stop Spark session ===
spark.stop()
