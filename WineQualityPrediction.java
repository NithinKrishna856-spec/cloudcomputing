import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WineQualityPrediction {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: WineQualityPrediction <path_to_input_csv>");
            System.exit(1);
        }

        String inputPath = args[0];
        String modelPath = "wine_quality_model";

        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction")
                .getOrCreate();

        Dataset<Row> input = spark.read().option("header", true)
                .option("inferSchema", true)
                .option("sep", ";")
                .csv(inputPath)
                .withColumnRenamed("quality", "label");

        PipelineModel model = PipelineModel.load(modelPath);

        Dataset<Row> predictions = model.transform(input);

        predictions.select("prediction", "label").show(10);

        double f1 = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1")
                .evaluate(predictions);

        System.out.println("✅ F1 Score on input: " + f1);

        spark.stop();
    }
}
