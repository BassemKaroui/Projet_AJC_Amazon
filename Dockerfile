FROM ubuntu

WORKDIR /root/data

RUN apt update \
    && apt upgrade -y \
    && apt install  openjdk-8-jre-headless -y \
    && apt install python3-pip -y \
    && pip3 install jupyter \
    && apt install wget -y \
    && wget https://archive.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz \
    && tar xzf spark-3.0.1-bin-hadoop2.7.tgz \
    && mv spark-3.0.1-bin-hadoop2.7 /opt/spark \
    && wget https://jdbc.postgresql.org/download/postgresql-42.2.12.jar \
    && mv postgresql-42.2.12.jar ..

ENV JUPYTER_HOME=/root/.local
ENV PATH=$JUPYTER_HOME/bin:$PATH
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON="jupyter"
ENV PYSPARK_DRIVER_PYTHON_OPTS="notebook --allow-root"
ENV PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH

ENV SPARK_CLASSPATH='/root/postgresql-42.2.12.jar'