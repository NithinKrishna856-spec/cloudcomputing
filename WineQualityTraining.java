import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class WineQualityTraining {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Training")
                .getOrCreate();

        // Load CSVs
        Dataset<Row> trainingData = spark.read().option("header", true)
                .option("inferSchema", true)
                .option("sep", ";")
                .csv("TrainingDataset.csv");

        Dataset<Row> validationData = spark.read().option("header", true)
                .option("inferSchema", true)
                .option("sep", ";")
                .csv("ValidationDataset.csv");

        // Prepare features
        String[] inputCols = trainingData.drop("quality").columns();
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        trainingData = trainingData.withColumnRenamed("quality", "label");
        validationData = validationData.withColumnRenamed("quality", "label");

        LogisticRegression lr = new LogisticRegression()
                .setFeaturesCol("features")
                .setLabelCol("label");

        Pipeline pipeline = new Pipeline().setStages(new org.apache.spark.ml.PipelineStage[]{assembler, lr});

        ParamGridBuilder paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[]{0.01, 0.1, 0.5})
                .addGrid(lr.elasticNetParam(), new double[]{0.0, 0.5, 1.0})
                .addGrid(lr.maxIter(), new int[]{10, 20});

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("label").setMetricName("f1"))
                .setEstimatorParamMaps(paramGrid.build())
                .setNumFolds(3);

        PipelineModel bestModel = cv.fit(trainingData);

        double f1 = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1")
                .evaluate(bestModel.transform(validationData));

        System.out.println("✅ F1 Score: " + f1);

        bestModel.write().overwrite().save("wine_quality_model");

        spark.stop();
    }
}
