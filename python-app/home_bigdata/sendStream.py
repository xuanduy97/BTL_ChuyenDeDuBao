#!/usr/bin/env python3
"""
Kafka Producer - Send home energy data from CSV to Kafka topic
"""

import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
import argparse
from datetime import datetime

class EnergyDataProducer:
    """Producer to send energy data to Kafka"""
    
    def __init__(self, kafka_server='localhost:9092', topic='BTLCDDDEnergy'):
        self.topic = topic
        self.producer = None
        self.kafka_server = kafka_server
        
        # Connect to Kafka
        self._connect()
    
    def _connect(self):
        """Connect to Kafka broker"""
        try:
            print(f"🔌 Connecting to Kafka at {self.kafka_server}...")
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_server],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            print(f"✅ Connected to Kafka!")
        except Exception as e:
            print(f"❌ Failed to connect to Kafka: {e}")
            raise
    
    def send_message(self, message, key=None):
        """Send a single message to Kafka"""
        try:
            future = self.producer.send(self.topic, value=message, key=key)
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            return True
        except KafkaError as e:
            print(f"❌ Failed to send message: {e}")
            return False
    
    def send_from_csv(self, csv_file, delay=0.1, limit=None, add_weather=False):
        """Send data from CSV file to Kafka"""
        print(f"\n{'='*80}")
        print(f"📂 Loading data from {csv_file}")
        print(f"{'='*80}")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return
        
        # Limit records if specified
        if limit and limit < len(df):
            df = df.head(limit)
            print(f"📊 Limiting to {limit} records")
        
        # Add weather data if needed (dummy data)
        if add_weather and 'temperature' not in df.columns:
            print("🌤️ Adding dummy weather data...")
            df['temperature'] = 20.0 + (df['use [kW]'] * 5)  # Dummy correlation
            df['humidity'] = 60.0 + (df['use [kW]'] * 10)
            df['pressure'] = 1013.0 + (df['use [kW]'] * 2)
            df['windSpeed'] = 5.0 + (df['use [kW]'] * 3)
        
        # Add data_type field
        df['data_type'] = 'power'
        
        print(f"\n{'='*80}")
        print(f"📡 Starting to send data to Kafka topic: {self.topic}")
        print(f"⏱️ Delay between messages: {delay}s")
        print(f"{'='*80}\n")
        
        sent_count = 0
        failed_count = 0
        start_time = time.time()
        
        try:
            for idx, row in df.iterrows():
                # Convert row to dictionary
                message = row.to_dict()
                
                # Convert numpy types to native Python types
                for key, value in message.items():
                    if pd.isna(value):
                        message[key] = 0.0
                    elif hasattr(value, 'item'):  # numpy type
                        message[key] = value.item()
                
                # Send to Kafka
                success = self.send_message(message, key=str(idx))
                
                if success:
                    sent_count += 1
                    
                    # Print progress every 100 messages
                    if sent_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = sent_count / elapsed
                        print(f"📤 Sent: {sent_count}/{len(df)} messages "
                              f"({sent_count/len(df)*100:.1f}%) - "
                              f"Rate: {rate:.1f} msg/s")
                else:
                    failed_count += 1
                
                # Delay between messages
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successfully sent: {sent_count}/{len(df)} messages")
        print(f"❌ Failed: {failed_count}")
        print(f"⏱️ Total time: {elapsed:.2f}s")
        print(f"📈 Average rate: {sent_count/elapsed:.2f} msg/s")
        print(f"{'='*80}\n")
    
    def send_continuous(self, csv_file, delay=1.0, loop=True):
        """Send data continuously (loop through CSV)"""
        print(f"\n🔄 Continuous mode - Will loop through data")
        print(f"💡 Press Ctrl+C to stop\n")
        
        loop_count = 0
        
        try:
            while True:
                loop_count += 1
                print(f"\n{'='*80}")
                print(f"🔄 LOOP #{loop_count}")
                print(f"{'='*80}")
                
                self.send_from_csv(csv_file, delay=delay, add_weather=True)
                
                if not loop:
                    break
                
                print(f"⏳ Waiting 5 seconds before next loop...\n")
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
    
    def close(self):
        """Close Kafka producer"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            print("✅ Producer closed")


def detect_kafka_server():
    """Auto-detect Kafka server address"""
    import socket
    
    # Check if running in Docker
    if os.path.exists('/.dockerenv'):
        print("🐳 Detected: Running inside Docker container")
        return 'kafka:9092'
    
    # Check if kafka hostname resolves (Docker network)
    try:
        socket.gethostbyname('kafka')
        print("🐳 Detected: Docker network available")
        return 'kafka:9092'
    except socket.gaierror:
        pass
    
    # Default to localhost
    print("💻 Detected: Running on host machine")
    return 'localhost:9092'


def main():
    """Main function"""
    # Auto-detect Kafka server
    default_kafka = detect_kafka_server()
    
    parser = argparse.ArgumentParser(description='Send home energy data to Kafka')
    parser.add_argument('--csv', type=str, default='home_data.csv',
                       help='Path to CSV file (default: home_data.csv)')
    parser.add_argument('--kafka', type=str, default=default_kafka,
                       help=f'Kafka server address (default: {default_kafka})')
    parser.add_argument('--topic', type=str, default='BTLCDDDEnergy',
                       help='Kafka topic name (default: BTLCDDDEnergy)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between messages in seconds (default: 0.1)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records to send (default: all)')
    parser.add_argument('--loop', action='store_true',
                       help='Loop continuously through data')
    parser.add_argument('--weather', action='store_true',
                       help='Add dummy weather data')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 KAFKA ENERGY DATA PRODUCER")
    print("="*80)
    print(f"📂 CSV File: {args.csv}")
    print(f"📡 Kafka Server: {args.kafka}")
    print(f"📮 Topic: {args.topic}")
    print(f"⏱️ Delay: {args.delay}s")
    print(f"🔄 Loop: {args.loop}")
    print(f"🌤️ Add Weather: {args.weather}")
    print("="*80)
    
    producer = None
    
    try:
        # Create producer
        producer = EnergyDataProducer(kafka_server=args.kafka, topic=args.topic)
        
        # Send data
        if args.loop:
            producer.send_continuous(args.csv, delay=args.delay)
        else:
            producer.send_from_csv(args.csv, delay=args.delay, 
                                  limit=args.limit, add_weather=args.weather)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if producer:
            producer.close()


if __name__ == "__main__":
    main()