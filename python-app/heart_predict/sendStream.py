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
    
    def load_csv(self, csv_file_path):
        """
        Đọc file CSV không xử lý gì
        
        Args:
            csv_file_path (str): Đường dẫn đến file CSV
            
        Returns:
            pandas.DataFrame: DataFrame gốc
        """
        try:
            # Đọc file CSV
            df = pd.read_csv(csv_file_path)
            logger.info(f"Đã đọc {len(df)} dòng từ file {csv_file_path}")
            logger.info(f"Các cột: {list(df.columns)}")
            logger.info(f"Kích thước dữ liệu: {df.shape}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file: {csv_file_path}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi đọc file CSV: {str(e)}")
            raise
    
    def send_data_alternating(self, df, batch_size=100, drop_list=['dataset', 'num'], delay_between_batches=1):
        """
        Gửi dữ liệu theo pattern:
        - Mỗi batch dữ liệu sẽ được gửi 2 lần:
          + Lần 1: chỉ xóa cột 'dataset' 
          + Lần 2: xóa theo drop_list (cùng dữ liệu)
        - Pair ID tăng dần cho mỗi batch dữ liệu mới
        
        Args:
            df (pandas.DataFrame): DataFrame chứa dữ liệu cần gửi
            batch_size (int): Số lượng record gửi trong mỗi batch
            drop_list (list): Danh sách cột cần xóa ở lần gửi thứ 1
            delay_between_batches (float): Thời gian nghỉ giữa các batch (giây)
        """
        total_records = len(df)
        sent_records = 0
        pair_id = 1
        
        logger.info(f"Bắt đầu gửi {total_records} records lên topic '{self.topic_name}'")
        logger.info(f"Pattern: Mỗi batch gửi 2 lần - Lần 1 xóa {drop_list}, Lần 2 xóa ['dataset']")
        
        # Gửi dữ liệu theo batch
        for i in range(0, total_records, batch_size):
            batch_end = min(i + batch_size, total_records)
            original_batch_df = df.iloc[i:batch_end].copy()
            
            logger.info(f"=== Pair ID {pair_id} - Xử lý {len(original_batch_df)} records ===")
            
            # Lần 1: Gửi batch chỉ xóa 'dataset'
            batch_df_1 = original_batch_df.copy()
            for col in drop_list:
                if col in batch_df_1.columns:
                    batch_df_1 = batch_df_1.drop(col, axis=1)
            
            logger.info(f"Lần 1 - Pair ID {pair_id} (xóa {drop_list}): Các cột còn lại: {list(batch_df_1.columns)}")
            
            batch_sent_1 = 0
            for index, row in batch_df_1.iterrows():
                try:
                    record_data = row.to_dict()
                    message_key = f"pair_{pair_id}_v1_{index}"
                    
                    future = self.producer.send(
                        topic=self.topic_name,
                        key=message_key,
                        value=record_data
                    )
                    
                    future.add_callback(self._on_send_success)
                    future.add_errback(self._on_send_error)
                    
                    sent_records += 1
                    batch_sent_1 += 1
                    
                except Exception as e:
                    logger.error(f"Lỗi khi gửi record {index} (lần 1): {str(e)}")
            
            self.producer.flush()
            logger.info(f"Đã gửi lần 1 - Pair ID {pair_id}: {batch_sent_1} records")
            
            # Nghỉ giữa 2 lần gửi
            time.sleep(delay_between_batches)
            
            # Lần 2: Gửi cùng batch nhưng xóa theo drop_list
            batch_df_2 = original_batch_df.copy()
            if 'dataset' in batch_df_2.columns:
                batch_df_2 = batch_df_2.drop('dataset', axis=1)
            
            logger.info(f"Lần 2 - Pair ID {pair_id} (chỉ xóa 'dataset'): Các cột còn lại: {list(batch_df_2.columns)}")
            
            batch_sent_2 = 0
            for index, row in batch_df_2.iterrows():
                try:
                    record_data = row.to_dict()
                    message_key = f"pair_{pair_id}_v2_{index}"
                    
                    future = self.producer.send(
                        topic=self.topic_name,
                        key=message_key,
                        value=record_data
                    )
                    
                    future.add_callback(self._on_send_success)
                    future.add_errback(self._on_send_error)
                    
                    sent_records += 1
                    batch_sent_2 += 1
                    
                except Exception as e:
                    logger.error(f"Lỗi khi gửi record {index} (lần 2): {str(e)}")
            
            self.producer.flush()
            logger.info(f"Đã gửi lần 2 - Pair ID {pair_id}: {batch_sent_2} records")
            logger.info(f"Hoàn thành Pair ID {pair_id} - Tổng cộng đã gửi: {sent_records} records")
            
            pair_id += 1
            
            # Nghỉ trước khi chuyển sang pair tiếp theo
            if batch_end < total_records:
                time.sleep(delay_between_batches)
        
        logger.info(f"Hoàn thành gửi tất cả - Tổng cộng: {sent_records} records từ {total_records} records gốc")
    
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
    DROP_LIST = ['dataset', 'num']  # Danh sách cột cần xóa trong batch lẻ
    
    # Khởi tạo producer
    producer = CSVKafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        topic_name=TOPIC_NAME
    )
    
    try:
        # Đọc CSV (không xử lý gì)
        df = producer.load_csv(CSV_FILE_PATH)
        
        # Gửi dữ liệu lên Kafka theo pattern xen kẽ
        producer.send_data_alternating(
            df=df,
            batch_size=BATCH_SIZE,
            drop_list=DROP_LIST,
            delay_between_batches=DELAY_BETWEEN_BATCHES
        )
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {str(e)}")
        
    finally:
        # Đóng producer
        producer.close()


if __name__ == "__main__":
    main()