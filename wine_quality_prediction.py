import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if len(sys.argv) != 2:
    print("Usage: spark-submit wine_quality_prediction.py <input_data_path>")
    sys.exit(1)

input_data_path = sys.argv[1]
model_path = "/home/jovyan/wine_quality_model"

# Start Spark session
spark = SparkSession.builder \
    .appName("Wine Quality Prediction") \
    .getOrCreate()

try:
    print(f"📦 Loading model from: {model_path}")
    model = PipelineModel.load(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    spark.stop()
    sys.exit(1)

try:
    print(f"📂 Loading input data from: {input_data_path}")
    input_data = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("sep", ";") \
        .option("quote", "\"") \
        .load(input_data_path)
except Exception as e:
    print(f"❌ Failed to load input data: {str(e)}")
    spark.stop()
    sys.exit(1)

# Clean column names
def clean_column(col_name):
    return col_name.replace('"', '').replace("'", "").strip()

input_data = input_data.toDF(*[clean_column(c) for c in input_data.columns])
print("🧹 Cleaned Columns:", input_data.columns)

if "quality" in input_data.columns:
    input_data = input_data.withColumn("label", input_data["quality"].cast("integer"))
    print("✅ 'quality' column found and converted to 'label'")
else:
    print("⚠️ 'quality' column not found in input data")

# Validate feature columns
feature_cols = model.stages[0].getInputCols()
missing_cols = [col for col in feature_cols if col not in input_data.columns]
if missing_cols:
    print(f"❌ Missing required columns: {missing_cols}")
    spark.stop()
    sys.exit(1)

# Run prediction
print("🔮 Running predictions...")
predictions = model.transform(input_data)

# Show a sample
print("🧾 Sample Predictions:")
predictions.select("prediction", "label").show(10)

# Evaluate
if "label" in predictions.columns:
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator.evaluate(predictions)
    print(f"\n✅ F1 Score on input data: {f1}")
else:
    print("⚠️ 'label' column not found. Skipping evaluation.")

spark.stop()
