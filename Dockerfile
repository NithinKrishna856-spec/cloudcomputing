FROM openjdk:11

# Install Spark
RUN apt-get update && apt-get install -y wget curl python3 && \
    wget https://dlcdn.apache.org/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz && \
    tar xvf spark-3.4.1-bin-hadoop3.tgz && \
    mv spark-3.4.1-bin-hadoop3 /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

WORKDIR /app

COPY WineQualityTraining.java .
COPY WineQualityPrediction.java .

# Compile the Java files
RUN javac -cp "$SPARK_HOME/jars/*" WineQualityTraining.java WineQualityPrediction.java

# Default to running prediction
ENTRYPOINT ["spark-submit", "--class", "WineQualityPrediction", "--master", "local", "WineQualityPrediction"]
