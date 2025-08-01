import pandas as pd
import json
from kafka import KafkaProducer
import time
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVKafkaProducer:
    def __init__(self, bootstrap_servers=['localhost:9092'], topic_name='heart_data'):
        """
        Khởi tạo Kafka Producer
        
        Args:
            bootstrap_servers (list): Danh sách Kafka broker servers
            topic_name (str): Tên topic để gửi dữ liệu
        """
        self.topic_name = topic_name
        
        # Cấu hình Kafka Producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            acks='all',  # Đảm bảo tất cả replica đều nhận được message
            retries=3,   # Số lần thử lại khi gửi thất bại
            batch_size=16384,  # Kích thước batch
            linger_ms=10       # Thời gian chờ trước khi gửi batch
        )
    
    def load_and_process_csv(self, csv_file_path):
        """
        Đọc file CSV và loại bỏ cột 'dataset'
        
        Args:
            csv_file_path (str): Đường dẫn đến file CSV
            
        Returns:
            pandas.DataFrame: DataFrame đã được xử lý
        """
        try:
            # Đọc file CSV
            df = pd.read_csv(csv_file_path)
            logger.info(f"Đã đọc {len(df)} dòng từ file {csv_file_path}")
            
            # Loại bỏ cột 'dataset' nếu tồn tại
            if 'dataset' in df.columns:
                df = df.drop('dataset', axis=1)
                logger.info("Đã loại bỏ cột 'dataset'")
            
            # Hiển thị thông tin về DataFrame
            logger.info(f"Các cột còn lại: {list(df.columns)}")
            logger.info(f"Kích thước dữ liệu: {df.shape}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file: {csv_file_path}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi đọc file CSV: {str(e)}")
            raise
    
    def send_data(self, df, batch_size=100, delay_between_batches=1):
        """
        Gửi dữ liệu từ DataFrame lên Kafka
        
        Args:
            df (pandas.DataFrame): DataFrame chứa dữ liệu cần gửi
            batch_size (int): Số lượng record gửi trong mỗi batch
            delay_between_batches (float): Thời gian nghỉ giữa các batch (giây)
        """
        total_records = len(df)
        sent_records = 0
        
        logger.info(f"Bắt đầu gửi {total_records} records lên topic '{self.topic_name}'")
        
        # Gửi dữ liệu theo batch
        for i in range(0, total_records, batch_size):
            batch_end = min(i + batch_size, total_records)
            batch_df = df.iloc[i:batch_end]
            
            # Gửi từng record trong batch
            for index, row in batch_df.iterrows():
                try:
                    # Chuyển đổi row thành dictionary
                    record_data = row.to_dict()
                    
                    # Tạo key cho message (sử dụng id nếu có, hoặc index)
                    message_key = str(record_data.get('id', index))
                    
                    # Gửi message lên Kafka
                    future = self.producer.send(
                        topic=self.topic_name,
                        key=message_key,
                        value=record_data
                    )
                    
                    # Callback xử lý kết quả gửi
                    future.add_callback(self._on_send_success)
                    future.add_errback(self._on_send_error)
                    
                    sent_records += 1
                    
                except Exception as e:
                    logger.error(f"Lỗi khi gửi record {index}: {str(e)}")
            
            # Flush để đảm bảo dữ liệu được gửi
            self.producer.flush()
            
            logger.info(f"Đã gửi batch {i//batch_size + 1}: {sent_records}/{total_records} records")
            
            # Nghỉ giữa các batch
            if batch_end < total_records:
                time.sleep(delay_between_batches)
        
        logger.info(f"Hoàn thành gửi {sent_records}/{total_records} records")
    
    def _on_send_success(self, record_metadata):
        """Callback khi gửi thành công"""
        logger.debug(f"Message gửi thành công: topic={record_metadata.topic}, "
                    f"partition={record_metadata.partition}, offset={record_metadata.offset}")
    
    def _on_send_error(self, excp):
        """Callback khi gửi thất bại"""
        logger.error(f"Lỗi khi gửi message: {str(excp)}")
    
    def close(self):
        """Đóng Kafka Producer"""
        if self.producer:
            self.producer.close()
            logger.info("Đã đóng Kafka Producer")


def main():
    """Hàm main để chạy chương trình"""
    # Cấu hình
    CSV_FILE_PATH = "heart_disease_uci.csv"  # Đường dẫn đến file CSV
    KAFKA_SERVERS = ['kafka:9092']  # Danh sách Kafka brokers
    TOPIC_NAME = "BTLCDDDHeart"  # Tên topic
    BATCH_SIZE = 1  # Số record gửi mỗi lần
    DELAY_BETWEEN_BATCHES = 1  # Thời gian nghỉ giữa các batch (giây)
    
    # Khởi tạo producer
    producer = CSVKafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        topic_name=TOPIC_NAME
    )
    
    try:
        # Đọc và xử lý CSV
        df = producer.load_and_process_csv(CSV_FILE_PATH)
        
        # Gửi dữ liệu lên Kafka
        producer.send_data(
            df=df,
            batch_size=BATCH_SIZE,
            delay_between_batches=DELAY_BETWEEN_BATCHES
        )
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {str(e)}")
        
    finally:
        # Đóng producer
        producer.close()


if __name__ == "__main__":
    main()