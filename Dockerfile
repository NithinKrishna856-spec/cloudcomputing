FROM jupyter/pyspark-notebook:latest

WORKDIR /app

# Install Python dependencies (using conda)
RUN conda install -y pandas scikit-learn boto3 && \
    conda clean -afy

# Install Hadoop AWS and AWS Java SDK JARs into Spark
USER root
ENV SPARK_HOME=/usr/local/spark

# Download JARs into the Spark classpath
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar $SPARK_HOME/jars/
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar $SPARK_HOME/jars/

USER jovyan

# Copy scripts
COPY wine_quality_training.py /app/
COPY wine_quality_prediction.py /app/

# Set entry point to the prediction script
ENTRYPOINT ["spark-submit", "/app/wine_quality_prediction.py"]
