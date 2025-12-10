import os
import sys
import json
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, DoubleType, IntegerType, StringType, StructField
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use TkAgg for real-time plotting
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import warnings
warnings.filterwarnings('ignore')

# ===== ATTENTION LAYER =====
class AttentionLayer(layers.Layer):
    """Custom Attention Layer"""
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W1 = self.add_weight(
            name='W1',
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b1 = self.add_weight(
            name='b1',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.W2 = self.add_weight(
            name='W2',
            shape=(input_shape[1][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b2 = self.add_weight(
            name='b2',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.V = self.add_weight(
            name='V',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
    def call(self, inputs):
        query, values = inputs
        query_with_time = tf.expand_dims(query, 1)
        score1 = tf.matmul(query_with_time, self.W1) + self.b1
        score2 = tf.matmul(values, self.W2) + self.b2
        score = tf.matmul(tf.nn.tanh(score1 + score2), self.V)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


# ===== MODEL BUILDER =====
def build_lstm_attention_model(time_steps, n_features, lstm_units=64, attention_units=32):
    """Build LSTM model with Attention"""
    inputs = layers.Input(shape=(time_steps, n_features))
    
    lstm_out = layers.LSTM(lstm_units, return_sequences=True, name='lstm_1')(inputs)
    lstm_out = layers.Dropout(0.2, name='dropout_1')(lstm_out)
    
    lstm_out2 = layers.LSTM(lstm_units, return_sequences=True, name='lstm_2')(lstm_out)
    lstm_out2 = layers.Dropout(0.2, name='dropout_2')(lstm_out2)
    
    query = layers.LSTM(lstm_units, return_sequences=False, name='lstm_3')(lstm_out2)
    
    attention = AttentionLayer(attention_units, name='attention')
    context_vector = attention([query, lstm_out2])
    
    combined = layers.Concatenate(name='concat')([query, context_vector])
    
    dense1 = layers.Dense(64, activation='relu', name='dense_1')(combined)
    dense1 = layers.Dropout(0.2, name='dropout_3')(dense1)
    
    dense2 = layers.Dense(32, activation='relu', name='dense_2')(dense1)
    outputs = layers.Dense(1, name='output')(dense2)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_model_from_weights(weights_path='model.weights.h5', config_path='model_config.json'):
    """Load model from weights and config"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = build_lstm_attention_model(
        time_steps=config['time_steps'],
        n_features=config['n_features'],
        lstm_units=config['lstm_units'],
        attention_units=config['attention_units']
    )
    
    model.load_weights(weights_path)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    
    return model


def save_model_config(model, filepath='model_config.json'):
    """Save model configuration"""
    config = {
        'time_steps': model.input_shape[1],
        'n_features': model.input_shape[2],
        'lstm_units': 64,
        'attention_units': 32
    }
    with open(filepath, 'w') as f:
        json.dump(config, f)
    print(f"✅ Config saved: {filepath}")


# ===== DATA PREPROCESSOR =====
class DataPreprocessor:
    """Data preprocessing for time series prediction"""
    
    def __init__(self, time_steps=60):
        self.scaler = StandardScaler()
        self.time_steps = time_steps
        self.feature_names = []
        self.data_buffer = []
        self.training_data = []  # Store data for retraining
        
    def get_feature_columns(self):
        """Get feature columns for home energy data"""
        return [
            'use [kW]', 'gen [kW]', 'House overall [kW]',
            'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
            'Home office [kW]', 'Fridge [kW]', 'temperature',
            'humidity', 'pressure', 'windSpeed'
        ]
    
    def load_scaler(self, scaler_path):
        """Load pre-trained scaler"""
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from {scaler_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            return False
    
    def save_scaler(self, scaler_path):
        """Save scaler"""
        try:
            joblib.dump(self.scaler, scaler_path)
            print(f"✅ Scaler saved to {scaler_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving scaler: {e}")
            return False
    
    def fit_scaler(self, data_df):
        """Fit scaler on data"""
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in data_df.columns]
        
        data = data_df[available_cols].values
        self.scaler.fit(data)
        self.feature_names = available_cols
        print(f"✅ Scaler fitted on {len(data)} samples")
    
    def add_to_buffer(self, data_dict):
        """Add new data point to buffer"""
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in data_dict]
        
        data_point = [data_dict.get(col, 0.0) for col in available_cols]
        self.data_buffer.append(data_point)
        self.training_data.append(data_point)
        
        if len(self.data_buffer) > self.time_steps + 10:
            self.data_buffer = self.data_buffer[-(self.time_steps + 10):]
        
        if not self.feature_names:
            self.feature_names = available_cols
    
    def prepare_sequence(self):
        """Prepare sequence for prediction"""
        if len(self.data_buffer) < self.time_steps:
            return None
        
        sequence = np.array(self.data_buffer[-self.time_steps:])
        sequence_scaled = self.scaler.transform(sequence)
        X = sequence_scaled.reshape(1, self.time_steps, -1)
        
        return X
    
    def prepare_training_data(self):
        """Prepare all accumulated data for training"""
        if len(self.training_data) < self.time_steps + 1:
            return None, None
        
        data = np.array(self.training_data)
        data_scaled = self.scaler.transform(data)
        
        n_sequences = len(data_scaled) - self.time_steps
        n_features = data_scaled.shape[1]
        
        X = np.zeros((n_sequences, self.time_steps, n_features), dtype=np.float32)
        y = np.zeros(n_sequences, dtype=np.float32)
        
        target_idx = 0  # 'use [kW]' is first column
        
        for i in range(n_sequences):
            X[i] = data_scaled[i:i+self.time_steps]
            y[i] = data_scaled[i+self.time_steps, target_idx]
        
        return X, y


# ===== LSTM PREDICTOR =====
class LSTMPredictor:
    """LSTM-based time series predictor"""
    
    def __init__(self, time_steps=60, n_features=12):
        self.model = None
        self.is_trained = False
        self.time_steps = time_steps
        self.n_features = n_features
        self.training_history = []
    
    def create_initial_model(self, csv_file, preprocessor, epochs=30, batch_size=32):
        """Create initial model from CSV file"""
        print("\n" + "="*80)
        print("🏗️ CREATING INITIAL MODEL")
        print("="*80)
        
        # Load data
        print(f"📂 Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} rows")
        
        # Limit data for faster training
        max_samples = 10000
        if len(df) > max_samples:
            df = df.iloc[-max_samples:]
            print(f"📊 Using last {max_samples} samples for initial training")
        
        # Prepare data
        feature_cols = preprocessor.get_feature_columns()
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Fit scaler
        preprocessor.fit_scaler(df)
        data_scaled = preprocessor.scaler.transform(df[available_cols].values)
        
        # Create sequences
        n_sequences = len(data_scaled) - self.time_steps
        X = np.zeros((n_sequences, self.time_steps, len(available_cols)), dtype=np.float32)
        y = np.zeros(n_sequences, dtype=np.float32)
        
        for i in range(n_sequences):
            X[i] = data_scaled[i:i+self.time_steps]
            y[i] = data_scaled[i+self.time_steps, 0]
        
        print(f"✅ Created {n_sequences} sequences")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Build model
        print("\n🏗️ Building model architecture...")
        self.model = build_lstm_attention_model(
            time_steps=self.time_steps,
            n_features=len(available_cols),
            lstm_units=64,
            attention_units=32
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )
        
        # Train
        print(f"\n🚀 Training model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                )
            ]
        )
        
        self.is_trained = True
        self.training_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'samples': len(X_train),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        })
        
        print(f"\n✅ Initial model trained successfully!")
        print(f"   Training Loss: {history.history['loss'][-1]:.6f}")
        print(f"   Validation Loss: {history.history['val_loss'][-1]:.6f}")
        
        return True
    
    def load_model(self, weights_path, config_path):
        """Load trained model"""
        try:
            self.model = load_model_from_weights(weights_path, config_path)
            self.is_trained = True
            print(f"✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def save_model(self, weights_path, config_path):
        """Save model"""
        try:
            self.model.save_weights(weights_path)
            save_model_config(self.model, config_path)
            print(f"✅ Model saved to {weights_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def predict_next(self, X):
        """Predict next value"""
        if not self.is_trained:
            return None
        
        try:
            prediction = self.model.predict(X, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def retrain(self, X, y, epochs=5):
        """Retrain model with new data"""
        if not self.is_trained or X is None or y is None:
            return False
        
        try:
            print(f"\n🔄 Retraining model with {len(X)} new samples...")
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
            
            self.training_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'samples': len(X),
                'final_loss': float(history.history['loss'][-1])
            })
            
            print(f"✅ Retraining completed! Loss: {history.history['loss'][-1]:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ Retraining error: {e}")
            return False


# ===== VISUALIZATION =====
class RealtimeVisualizer:
    """Real-time visualization of predictions"""
    
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.actual = deque(maxlen=max_points)
        self.predicted = deque(maxlen=max_points)
        self.errors = deque(maxlen=max_points)
        
        # Setup plot
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Real-time Energy Prediction', fontsize=16, fontweight='bold')
        
        # Initialize lines
        self.line_actual, = self.ax1.plot([], [], 'b-', label='Actual', linewidth=2)
        self.line_pred, = self.ax1.plot([], [], 'r--', label='Predicted', linewidth=2)
        self.line_error, = self.ax2.plot([], [], 'g-', label='Error', linewidth=2)
        
        self.ax1.set_ylabel('Energy (kW)', fontsize=12)
        self.ax1.set_title('Energy Consumption: Actual vs Predicted')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_xlabel('Time Step', fontsize=12)
        self.ax2.set_ylabel('Absolute Error (kW)', fontsize=12)
        self.ax2.set_title('Prediction Error')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update(self, actual_val, predicted_val):
        """Update plot with new data"""
        self.times.append(len(self.times))
        self.actual.append(actual_val)
        self.predicted.append(predicted_val)
        self.errors.append(abs(actual_val - predicted_val))
        
        # Update data
        self.line_actual.set_data(list(self.times), list(self.actual))
        self.line_pred.set_data(list(self.times), list(self.predicted))
        self.line_error.set_data(list(self.times), list(self.errors))
        
        # Rescale axes
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        """Close plot"""
        plt.close(self.fig)


# ===== REPORT GENERATOR =====
class ReportGenerator:
    """Generate performance report"""
    
    def __init__(self):
        self.predictions = []
        self.start_time = datetime.now()
    
    def add_prediction(self, timestamp, actual, predicted, error):
        """Add prediction record"""
        self.predictions.append({
            'timestamp': timestamp,
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'error_pct': (error / actual * 100) if actual > 0 else 0
        })
    
    def generate_report(self, output_file='prediction_report.txt'):
        """Generate and save report"""
        if not self.predictions:
            return
        
        df = pd.DataFrame(self.predictions)
        
        # Calculate metrics
        mae = df['error'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        mape = df['error_pct'].mean()
        max_error = df['error'].max()
        min_error = df['error'].min()
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("ENERGY PREDICTION PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duration: {datetime.now() - self.start_time}")
        report.append(f"\nTotal Predictions: {len(self.predictions)}")
        report.append(f"\n{'METRICS':-^80}")
        report.append(f"Mean Absolute Error (MAE):     {mae:.6f} kW")
        report.append(f"Root Mean Square Error (RMSE): {rmse:.6f} kW")
        report.append(f"Mean Absolute % Error (MAPE):  {mape:.2f}%")
        report.append(f"Maximum Error:                 {max_error:.6f} kW")
        report.append(f"Minimum Error:                 {min_error:.6f} kW")
        
        report.append(f"\n{'STATISTICS':-^80}")
        report.append(f"Actual Consumption:")
        report.append(f"  Mean:   {df['actual'].mean():.6f} kW")
        report.append(f"  Median: {df['actual'].median():.6f} kW")
        report.append(f"  Std:    {df['actual'].std():.6f} kW")
        report.append(f"  Min:    {df['actual'].min():.6f} kW")
        report.append(f"  Max:    {df['actual'].max():.6f} kW")
        
        report.append(f"\nPredicted Consumption:")
        report.append(f"  Mean:   {df['predicted'].mean():.6f} kW")
        report.append(f"  Median: {df['predicted'].median():.6f} kW")
        report.append(f"  Std:    {df['predicted'].std():.6f} kW")
        report.append(f"  Min:    {df['predicted'].min():.6f} kW")
        report.append(f"  Max:    {df['predicted'].max():.6f} kW")
        
        report.append(f"\n{'RECENT PREDICTIONS (Last 10)':-^80}")
        report.append(f"{'Time':<20} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'Error %':>10}")
        report.append("-" * 80)
        
        for pred in self.predictions[-10:]:
            report.append(
                f"{pred['timestamp']:<20} "
                f"{pred['actual']:>10.4f} "
                f"{pred['predicted']:>10.4f} "
                f"{pred['error']:>10.4f} "
                f"{pred['error_pct']:>10.2f}%"
            )
        
        report.append("=" * 80)
        
        # Save to file
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        # Print to console
        print("\n" + report_text)
        print(f"\n✅ Report saved to {output_file}")


# ===== KAFKA STREAMING SYSTEM =====
class EnergyPredictionKafkaSystem:
    """Kafka streaming system with auto-training and visualization"""
    
    def __init__(self, 
                 topic_name='BTLCDDDEnergy',
                 kafka_server='kafka:9092',
                 time_steps=60,
                 csv_file='home_data.csv',
                 model_weights_path='model.weights.h5',
                 model_config_path='model_config.json',
                 scaler_path='scaler.pkl',
                 retrain_interval=100):
        
        self.topic_name = topic_name
        self.kafka_server = kafka_server
        self.csv_file = csv_file
        self.spark = None
        self.query = None
        
        # ML components
        self.preprocessor = DataPreprocessor(time_steps=time_steps)
        self.predictor = LSTMPredictor(time_steps=time_steps)
        
        # Paths
        self.model_weights_path = model_weights_path
        self.model_config_path = model_config_path
        self.scaler_path = scaler_path
        
        # Tracking
        self.prediction_count = 0
        self.retrain_interval = retrain_interval
        self.last_retrain_count = 0
        
        # Visualization
        self.visualizer = None
        self.report_gen = ReportGenerator()
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or create model"""
        print("\n" + "="*80)
        print("🔍 CHECKING MODEL STATUS")
        print("="*80)
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_weights_path) or '.', exist_ok=True)
        
        # Check if all required files exist
        model_exists = os.path.exists(self.model_weights_path)
        config_exists = os.path.exists(self.model_config_path)
        scaler_exists = os.path.exists(self.scaler_path)
        
        print(f"📁 Looking for:")
        print(f"   Model weights: {self.model_weights_path} - {'✅' if model_exists else '❌'}")
        print(f"   Model config:  {self.model_config_path} - {'✅' if config_exists else '❌'}")
        print(f"   Scaler:        {self.scaler_path} - {'✅' if scaler_exists else '❌'}")
        
        if model_exists and config_exists and scaler_exists:
            print("\n✅ Found existing model files - Loading...")
            self.predictor.load_model(self.model_weights_path, self.model_config_path)
            self.preprocessor.load_scaler(self.scaler_path)
        else:
            print(f"\n❌ Model files not found")
            
            if os.path.exists(self.csv_file):
                print(f"\n📂 Found training data: {self.csv_file}")
                print(f"🏗️ Creating initial model (this may take a few minutes)...")
                
                success = self.predictor.create_initial_model(
                    self.csv_file, 
                    self.preprocessor,
                    epochs=30
                )
                
                if success:
                    # Save model and scaler
                    print(f"\n💾 Saving model to {os.path.dirname(self.model_weights_path)}/")
                    self.predictor.save_model(self.model_weights_path, self.model_config_path)
                    self.preprocessor.save_scaler(self.scaler_path)
                    print("✅ Model creation complete!")
                else:
                    print("❌ Failed to create initial model")
            else:
                print(f"\n⚠️ CSV file not found: {self.csv_file}")
                print("💡 Please provide training data in one of these ways:")
                print(f"   1. Place CSV file at: {self.csv_file}")
                print("   2. System will collect data from Kafka stream")
                print("   3. After collecting enough data, model will be auto-created")
                print("\n🔄 System will continue without model - waiting for data...")
    
    def create_spark_session(self):
        """Create Spark session"""
        scala_version = '2.12'
        spark_version = '3.5.5'
        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
            'org.apache.kafka:kafka-clients:3.9.1'
        ]
        
        self.spark = SparkSession.builder \
            .appName("EnergyPredictionKafkaSystem") \
            .master("local[*]") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        print("✓ Spark session created")
    
    def get_schema(self):
        """Define schema"""
        return StructType([
            StructField("time", IntegerType(), True),
            StructField("use [kW]", DoubleType(), True),
            StructField("gen [kW]", DoubleType(), True),
            StructField("House overall [kW]", DoubleType(), True),
            StructField("Dishwasher [kW]", DoubleType(), True),
            StructField("Furnace 1 [kW]", DoubleType(), True),
            StructField("Furnace 2 [kW]", DoubleType(), True),
            StructField("Home office [kW]", DoubleType(), True),
            StructField("Fridge [kW]", DoubleType(), True),
            StructField("Wine cellar [kW]", DoubleType(), True),
            StructField("Garage door [kW]", DoubleType(), True),
            StructField("Kitchen 12 [kW]", DoubleType(), True),
            StructField("Kitchen 14 [kW]", DoubleType(), True),
            StructField("Kitchen 38 [kW]", DoubleType(), True),
            StructField("Barn [kW]", DoubleType(), True),
            StructField("Well [kW]", DoubleType(), True),
            StructField("Microwave [kW]", DoubleType(), True),
            StructField("Living room [kW]", DoubleType(), True),
            StructField("Solar [kW]", DoubleType(), True),
            StructField("data_type", StringType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("humidity", DoubleType(), True),
            StructField("pressure", DoubleType(), True),
            StructField("windSpeed", DoubleType(), True),
        ])
    
    def create_kafka_stream(self):
        """Create Kafka stream"""
        try:
            kafka_df = self.spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.kafka_server) \
                .option("subscribe", self.topic_name) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .load()
            return kafka_df
        except Exception as e:
            print(f"\n❌ Error creating Kafka stream: {e}")
            raise
    
    def parse_data(self, kafka_df):
        """Parse JSON data"""
        schema = self.get_schema()
        parsed_df = kafka_df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        return parsed_df
    
    def process_batch(self, df, epoch_id):
        """Process each batch"""
        if df.count() == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"⚡ BATCH {epoch_id} - Processing {df.count()} records")
        print(f"{'='*80}")
        
        if not self.predictor.is_trained:
            print("⚠️ Model not trained yet. Collecting data...")
            
            # Collect data for future training
            records = df.collect()
            for row in records:
                row_dict = row.asDict()
                self.preprocessor.add_to_buffer(row_dict)
            
            print(f"📊 Data collected: {len(self.preprocessor.training_data)} samples")
            print(f"💡 Need {self.preprocessor.time_steps + 100} samples minimum for training")
            
            # Try to create model if we have enough data
            if len(self.preprocessor.training_data) >= self.preprocessor.time_steps + 100:
                print(f"\n✅ Enough data collected! Creating model from stream data...")
                self._create_model_from_stream()
            
            print(f"{'='*80}\n")
            return
        
        records = df.collect()
        
        for row in records:
            row_dict = row.asDict()
            timestamp = datetime.fromtimestamp(row_dict.get('time', time.time())).strftime('%H:%M:%S')
            current_use = row_dict.get('use [kW]', 0.0)
            
            # Add to buffer
            self.preprocessor.add_to_buffer(row_dict)
            
            # Make prediction if buffer is ready
            if len(self.preprocessor.data_buffer) >= self.preprocessor.time_steps:
                X = self.preprocessor.prepare_sequence()
                
                if X is not None:
                    predicted_use = self.predictor.predict_next(X)
                    
                    if predicted_use is not None:
                        self.prediction_count += 1
                        error = abs(predicted_use - current_use)
                        error_pct = (error / current_use * 100) if current_use > 0 else 0
                        
                        # Add to report
                        self.report_gen.add_prediction(timestamp, current_use, predicted_use, error)
                        
                        # Update visualization
                        if self.visualizer:
                            self.visualizer.update(current_use, predicted_use)
                        
                        # Print prediction
                        print(f"\n🔮 PREDICTION #{self.prediction_count}")
                        print(f"   ⏰ Time: {timestamp}")
                        print(f"   📊 Actual: {current_use:.4f} kW")
                        print(f"   🎯 Predicted: {predicted_use:.4f} kW")
                        print(f"   📉 Error: {error:.4f} kW ({error_pct:.2f}%)")
                        
                        # Alert
                        if predicted_use > 1.5:
                            print(f"   🔴 HIGH consumption predicted!")
                        elif predicted_use > 1.0:
                            print(f"   🟡 Medium consumption")
                        else:
                            print(f"   🟢 Normal consumption")
                        
                        # Retrain check
                        if self.prediction_count - self.last_retrain_count >= self.retrain_interval:
                            self._retrain_model()
                            self.last_retrain_count = self.prediction_count
        
        print(f"{'='*80}\n")
    
    def _create_model_from_stream(self):
        """Create model from streaming data"""
        print("\n" + "="*80)
        print("🏗️ CREATING MODEL FROM STREAM DATA")
        print("="*80)
        
        # Prepare training data
        data = np.array(self.preprocessor.training_data)
        
        # Fit scaler
        print("📊 Fitting scaler...")
        self.preprocessor.scaler.fit(data)
        data_scaled = self.preprocessor.scaler.transform(data)
        
        # Create sequences
        n_sequences = len(data_scaled) - self.preprocessor.time_steps
        n_features = data_scaled.shape[1]
        
        X = np.zeros((n_sequences, self.preprocessor.time_steps, n_features), dtype=np.float32)
        y = np.zeros(n_sequences, dtype=np.float32)
        
        for i in range(n_sequences):
            X[i] = data_scaled[i:i+self.preprocessor.time_steps]
            y[i] = data_scaled[i+self.preprocessor.time_steps, 0]
        
        print(f"✅ Created {n_sequences} sequences")
        
        # Build model
        print("🏗️ Building model...")
        self.predictor.model = build_lstm_attention_model(
            time_steps=self.preprocessor.time_steps,
            n_features=n_features,
            lstm_units=64,
            attention_units=32
        )
        
        self.predictor.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )
        
        # Train
        print("🚀 Training model (20 epochs)...")
        history = self.predictor.model.fit(
            X, y,
            epochs=20,
            batch_size=32,
            verbose=1,
            validation_split=0.2
        )
        
        self.predictor.is_trained = True
        
        # Save
        print(f"\n💾 Saving model...")
        self.predictor.save_model(self.model_weights_path, self.model_config_path)
        self.preprocessor.save_scaler(self.scaler_path)
        
        print(f"✅ Model created and saved successfully!")
        print(f"   Final Loss: {history.history['loss'][-1]:.6f}")
        print("="*80 + "\n")
    
    def _retrain_model(self):
        """Retrain model with accumulated data"""
        print("\n" + "="*80)
        print("🔄 RETRAINING MODEL")
        print("="*80)
        
        X, y = self.preprocessor.prepare_training_data()
        
        if X is not None and y is not None:
            success = self.predictor.retrain(X, y, epochs=5)
            
            if success:
                # Save updated model
                self.predictor.save_model(self.model_weights_path, self.model_config_path)
                self.preprocessor.save_scaler(self.scaler_path)
                print("💾 Model saved after retraining")
        else:
            print("⚠️ Not enough data for retraining yet")
        
        print("="*80 + "\n")
    
    def start_stream(self, processing_time="5 seconds", enable_viz=True):
        """Start stream processing"""
        if not self.spark:
            raise Exception("Spark session not created")
        
        # Initialize visualization
        if enable_viz:
            self.visualizer = RealtimeVisualizer(max_points=100)
        
        try:
            kafka_df = self.create_kafka_stream()
            parsed_df = self.parse_data(kafka_df)
            power_df = parsed_df.filter(col("data_type") == "power")
            
            self.query = power_df.writeStream \
                .outputMode("append") \
                .option("checkpointLocation", "checkpoint/energy_prediction/") \
                .trigger(processingTime=processing_time) \
                .foreachBatch(self.process_batch) \
                .start()
            
            print("✓ Kafka stream started")
            return self.query
            
        except Exception as e:
            print(f"\n❌ Failed to start stream: {e}")
            raise
    
    def wait_for_termination(self):
        """Wait for stream termination"""
        if self.query:
            try:
                self.query.awaitTermination()
            except KeyboardInterrupt:
                print("\n⏹ Stream interrupted by user")
    
    def stop_stream(self):
        """Stop stream"""
        if self.query:
            self.query.stop()
            print("✓ Stream stopped")
    
    def stop_spark(self):
        """Stop Spark"""
        if self.spark:
            self.spark.stop()
            print("✓ Spark stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_stream()
        
        # Generate final report
        print("\n📊 Generating final report...")
        self.report_gen.generate_report('prediction_report.txt')
        
        # Close visualization
        if self.visualizer:
            self.visualizer.close()
        
        self.stop_spark()
        print("✓ Resources cleaned up")


# ===== MAIN =====
def main():
    """Main function"""
    system = None
    
    try:
        print("🚀 Starting Enhanced Energy Prediction System...")
        print(f"📦 TensorFlow: {tf.__version__}")
        
        # Create system
        system = EnergyPredictionKafkaSystem(
            topic_name='BTLCDDDEnergy',
            kafka_server='kafka:9092',
            time_steps=60,
            csv_file='home_data.csv',
            model_weights_path='models/model.weights.h5',
            model_config_path='models/model_config.json',
            scaler_path='models/scaler.pkl',
            retrain_interval=100  # Retrain every 100 predictions
        )
        
        # Create Spark session
        system.create_spark_session()
        
        # Start stream
        system.start_stream(processing_time="5 seconds", enable_viz=True)
        
        print("\n📊 System Status:")
        print(f"   🤖 Model trained: {system.predictor.is_trained}")
        print(f"   ⏱️ Time steps: {system.preprocessor.time_steps}")
        print(f"   🔄 Retrain interval: {system.retrain_interval} predictions")
        
        print("\n💡 Features:")
        print("   ⚡ Real-time predictions with LSTM-Attention")
        print("   📈 Auto-retraining every 100 predictions")
        print("   📊 Live visualization")
        print("   📝 Performance report generation")
        
        print("\n💡 Press Ctrl+C to stop and generate report...")
        print("-" * 60)
        
        # Wait
        system.wait_for_termination()
        
    except KeyboardInterrupt:
        print("\n⏹ Stopping system...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if system:
            system.cleanup()


if __name__ == "__main__":
    main()