FROM eclipse-temurin:17-jdk
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# tạo venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ========== 1. Cài Python, pip và các công cụ cần thiết ==========
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget curl && \
    rm -rf /var/lib/apt/lists/*

# ========== 2. Cài Spark ==========
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# ========== 3. Thiết lập biến môi trường ==========
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# ========== 4. Cài thư viện Python ==========
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ========== 5. Copy mã nguồn vào container ==========
WORKDIR /app
COPY . .

# ========== 6. Mặc định giữ container chạy ==========
CMD ["tail", "-f", "/dev/null"]
