FROM ubuntu

WORKDIR /root

RUN apt update \
    && apt upgrade -y \
    && apt install  openjdk-8-jre-headless -y \
    && apt install python3-pip -y \
    && pip3 install jupyter \
    && apt install wget -y \
    && wget https://archive.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz \
    && tar xzf spark-3.0.1-bin-hadoop2.7.tgz \
    && mv spark-3.0.1-bin-hadoop2.7 /opt/spark \
    && wget https://jdbc.postgresql.org/download/postgresql-42.2.12.jar

RUN pip3 install flask \
    && pip3 install sqlalchemy \
    && pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install transformers

ENV JUPYTER_HOME=/root/.local
ENV PATH=$JUPYTER_HOME/bin:$PATH
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON="jupyter"
ENV PYSPARK_DRIVER_PYTHON_OPTS="notebook --allow-root"
ENV PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH

ENV SPARK_CLASSPATH='/root/postgresql-42.2.12.jar'

EXPOSE 8888
EXPOSE 5000

WORKDIR /root/workspace