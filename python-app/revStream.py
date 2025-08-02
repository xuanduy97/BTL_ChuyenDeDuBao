import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, pandas_udf, lit
from pyspark.sql.types import StructType, DoubleType, IntegerType, StringType, StructField, BooleanType
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Check Spark availability
try:
    from pyspark.sql.functions import pandas_udf
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

class DataPreprocessor:
    """Data preprocessing class with encoding and scaling capabilities"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, csv_file_path):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(csv_file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def explore_data(self, data):
        """Basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {data.shape}")
        print(f"\nColumns: {list(data.columns)}")
        print(f"\nMissing values:\n{data.isnull().sum()}")
        print(f"\nData types:\n{data.dtypes}")
        
        if 'num' in data.columns:
            print(f"\nTarget distribution:\n{data['num'].value_counts()}")
    
    def encode_categorical_features(self, data):
        """Encode categorical features"""
        data_encoded = data.copy()
        categorical_columns = ['sex', 'cp', 'restecg', 'slope', 'thal']
        
        for column in categorical_columns:
            if column in data_encoded.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    # Handle missing values
                    data_encoded[column] = data_encoded[column].fillna('unknown').astype(str)
                    self.label_encoders[column].fit(data_encoded[column])
                
                data_encoded[column] = self.label_encoders[column].transform(
                    data_encoded[column].fillna('unknown').astype(str)
                )
        
        return data_encoded
    
    def prepare_features_target(self, data):
        """Prepare features and target"""
        # Drop non-predictive columns
        columns_to_drop = ['id'] if 'id' in data.columns else []
        
        if columns_to_drop:
            data = data.drop(columns_to_drop, axis=1)
        
        # Separate features and target
        if 'num' in data.columns:
            X = data.drop('num', axis=1)
            y = data['num']
        else:
            X = data
            y = None
        
        self.feature_names = list(X.columns)
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_names,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=self.feature_names,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled

class HeartDiseasePredictor:
    """Heart Disease Prediction Model with Spark UDF support"""
    
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        self.is_trained = False
        self.best_params = None
    
    def train_model(self, X_train, y_train, tune_hyperparameters=False):
        """Train Random Forest model"""
        print("\n=== TRAINING RANDOM FOREST MODEL ===")
        
        n_samples = len(X_train)
        if n_samples < 5:
            print(f"Warning: Only {n_samples} samples available.")
            tune_hyperparameters = False
        
        if tune_hyperparameters and n_samples >= 10:
            print("Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train, y_train)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Cross-validation score
        if n_samples >= 3:
            try:
                cv_folds = min(3, n_samples)
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Cross-validation failed: {e}")
        
        print(f"Model trained successfully with {n_samples} samples")
        return self.model
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        n_samples = len(X_train)
        cv_folds = min(3, max(2, n_samples // 3))
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt']
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        try:
            grid_search = GridSearchCV(
                rf, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def predict_single_patient(self, patient_data, preprocessor):
        """Predict heart disease for a single patient"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return None
        
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(patient_data, dict):
                patient_df = pd.DataFrame([patient_data])
            else:
                patient_df = patient_data.copy()
            
            # Remove id and num columns if they exist
            columns_to_drop = []
            if 'id' in patient_df.columns:
                columns_to_drop.append('id')
            if 'num' in patient_df.columns:
                columns_to_drop.append('num')
            
            if columns_to_drop:
                patient_df = patient_df.drop(columns_to_drop, axis=1)
            
            # Encode categorical features
            for column, encoder in preprocessor.label_encoders.items():
                if column in patient_df.columns:
                    try:
                        patient_df[column] = patient_df[column].fillna('unknown').astype(str)
                        patient_df[column] = encoder.transform(patient_df[column])
                    except ValueError:
                        # Handle unknown categories
                        patient_df[column] = encoder.transform([encoder.classes_[0]] * len(patient_df))
            
            # Ensure all required features are present
            for feature in preprocessor.feature_names:
                if feature not in patient_df.columns:
                    patient_df[feature] = 0
            
            # Select and order features correctly
            X_patient = patient_df[preprocessor.feature_names]
            
            # Scale features
            X_patient_scaled = preprocessor.scaler.transform(X_patient)
            
            # Make prediction
            prediction = self.model.predict(X_patient_scaled)[0]
            probabilities = self.model.predict_proba(X_patient_scaled)[0]
            
            result = {
                'prediction': int(prediction),
                'probability_no_disease': float(probabilities[0]),
                'probability_disease': float(probabilities[1]),
                'risk_level': 'High' if probabilities[1] > 0.7 else 'Medium' if probabilities[1] > 0.3 else 'Low'
            }
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def retrain_with_new_data(self, new_data_list, preprocessor, tune_hyperparameters=False):
        """Retrain model with new data"""
        try:
            print(f"\n=== RETRAINING WITH {len(new_data_list)} NEW SAMPLES ===")
            
            # Convert list of Row objects to DataFrame
            data_dicts = []
            for row in new_data_list:
                if hasattr(row, 'asDict'):
                    data_dicts.append(row.asDict())
                else:
                    data_dicts.append(row)
            
            new_df = pd.DataFrame(data_dicts)
            print(f"New data shape: {new_df.shape}")
            print(f"Target distribution: {new_df['num'].value_counts().to_dict()}")
            
            # Preprocess new data
            new_data_encoded = preprocessor.encode_categorical_features(new_df)
            X_new, y_new = preprocessor.prepare_features_target(new_data_encoded)
            
            # Scale features
            X_new_scaled = pd.DataFrame(
                preprocessor.scaler.transform(X_new),
                columns=preprocessor.feature_names
            )
            
            # Retrain model
            self.train_model(X_new_scaled, y_new, tune_hyperparameters)
            
            print("Retraining completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in retraining: {e}")
            return False
    
    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return False
        
        try:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class HeartDiseaseKafkaSystem:
    """Integrated system combining Kafka streaming with ML prediction and retraining"""
    
    def __init__(self, topic_name='BTLCDDDHeart', kafka_server='kafka:9092', model_path=None):
        self.topic_name = topic_name
        self.kafka_server = kafka_server
        self.spark = None
        self.query = None
        
        # ML components
        self.preprocessor = DataPreprocessor()
        self.predictor = HeartDiseasePredictor()
        
        # Data storage for retraining
        self.accumulated_valid_records = []
        self.retrain_batch_size = 5
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_system(model_path)
        else:
            print("No pre-trained model found. Will need initial training data.")
    
    def create_spark_session(self):
        """Create Spark session with Kafka configuration"""
        scala_version = '2.12'
        spark_version = '3.5.5'
        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
            'org.apache.kafka:kafka-clients:3.9.1'
        ]
        
        self.spark = SparkSession.builder \
            .appName("HeartDiseaseKafkaSystem") \
            .master("local[*]") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        print("✓ Spark session created")
    
    def get_heart_disease_schema(self):
        """Define schema for heart disease data"""
        return StructType([
            StructField("id", IntegerType(), True),
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("cp", StringType(), True),
            StructField("trestbps", DoubleType(), True),
            StructField("chol", DoubleType(), True),
            StructField("fbs", BooleanType(), True),
            StructField("restecg", StringType(), True),
            StructField("thalch", DoubleType(), True),
            StructField("exang", BooleanType(), True),
            StructField("oldpeak", DoubleType(), True),
            StructField("slope", StringType(), True),
            StructField("ca", DoubleType(), True),
            StructField("thal", StringType(), True),
            StructField("num", IntegerType(), True)
        ])
    
    def create_kafka_stream(self):
        """Create Kafka stream DataFrame"""
        kafka_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_server) \
            .option("subscribe", self.topic_name) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .load()
        
        return kafka_df
    
    def parse_data(self, kafka_df):
        """Parse JSON data from Kafka stream"""
        schema = self.get_heart_disease_schema()
        parsed_df = kafka_df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        return parsed_df
    
    def process_batch(self, df, epoch_id):
        """Process each batch with prediction and retraining logic"""
        if df.count() == 0:
            print(f"\n=== Batch {epoch_id} ===")
            print("Empty batch received")
            return
        
        total_count = df.count()
        
        # Separate null and non-null records
        null_records = df.filter(col("num").isNull())
        valid_records = df.filter(col("num").isNotNull())
        
        null_count = null_records.count()
        valid_count = valid_records.count()
        
        print(f"\n=== Batch {epoch_id} ===")
        print(f"📊 Total: {total_count} | Valid: {valid_count} | Null: {null_count}")
        
        # Process records with num=null for prediction
        if null_count > 0:
            self._process_prediction_records(null_records, null_count)
        
        # Accumulate valid records for retraining
        if valid_count > 0:
            self._process_training_records(valid_records, valid_count)
        
        print("-" * 60)
    
    def _process_prediction_records(self, null_records, null_count):
        """Process records with num=null for prediction"""
        print(f"\n🔮 PROCESSING {null_count} RECORDS FOR PREDICTION")
        print("-" * 80)
        
        if not self.predictor.is_trained:
            print("⚠️ Model not trained yet. Cannot make predictions.")
            print("💡 Waiting for training data (records with num≠null)...")
            print(f"   Need {self.retrain_batch_size} valid records to create initial model")
            print(f"   Currently have: {len(self.accumulated_valid_records)} valid records")
            
            # Show what we're waiting for
            if len(self.accumulated_valid_records) > 0:
                print("📋 Accumulated training records so far:")
                for i, row in enumerate(self.accumulated_valid_records):
                    row_dict = row.asDict() if hasattr(row, 'asDict') else row
                    print(f"   {i+1}. ID: {row_dict.get('id', 'N/A')}, Age: {row_dict.get('age', 'N/A')}, Target: {row_dict.get('num', 'N/A')}")
            
            return
        
        try:
            for i, row in enumerate(null_records.collect()):
                row_dict = row.asDict()
                patient_id = row_dict.get('id', f'unknown_{i}')
                
                print(f"\n🔍 Predicting for Patient ID: {patient_id}")
                
                # Make prediction
                result = self.predictor.predict_single_patient(row_dict, self.preprocessor)
                
                if result:
                    prediction_text = "Heart Disease" if result['prediction'] == 1 else "No Heart Disease"
                    print(f"   ✅ Prediction: {prediction_text}")
                    print(f"   📊 Disease Probability: {result['probability_disease']:.4f}")
                    print(f"   ⚡ Risk Level: {result['risk_level']}")
                else:
                    print(f"   ❌ Prediction failed for Patient ID: {patient_id}")
        
        except Exception as e:
            print(f"❌ Error in prediction processing: {e}")
        
        print("-" * 80)
    
    def _process_training_records(self, valid_records, valid_count):
        """Process valid records for training/retraining"""
        print(f"\n📚 PROCESSING {valid_count} VALID RECORDS FOR TRAINING")
        
        # Add to accumulated records
        current_valid = valid_records.collect()
        self.accumulated_valid_records.extend(current_valid)
        
        print(f"✅ Accumulated valid records: {len(self.accumulated_valid_records)}")
        
        # Check if we have enough records for training/retraining
        if len(self.accumulated_valid_records) >= self.retrain_batch_size:
            
            # Get batch for training/retraining
            batch_to_process = self.accumulated_valid_records[:self.retrain_batch_size]
            
            try:
                # Display batch info
                batch_df = self.spark.createDataFrame(batch_to_process)
                print("📋 Training batch preview:")
                batch_df.show(self.retrain_batch_size, truncate=False)
                
                print("\n📊 Target Distribution in Training Batch:")
                batch_df.groupBy("num").count().orderBy("num").show()
                
                # Check if model exists - if not, create initial model
                if not self.predictor.is_trained:
                    print(f"\n🆕 NO MODEL FOUND - CREATING INITIAL MODEL WITH {self.retrain_batch_size} RECORDS")
                    print("=" * 100)
                    
                    success = self._create_initial_model_from_stream(batch_to_process)
                    
                    if success:
                        print("🎉 Initial model created successfully from streaming data!")
                        # Save initial model
                        self.save_system("models/heart_disease_model_stream_initial.pkl")
                    else:
                        print("❌ Initial model creation failed!")
                else:
                    # Model exists, do retraining
                    print(f"\n🔄 RETRAINING EXISTING MODEL WITH {self.retrain_batch_size} RECORDS")
                    print("=" * 100)
                    
                    retrain_success = self.predictor.retrain_with_new_data(
                        batch_to_process, 
                        self.preprocessor, 
                        tune_hyperparameters=False
                    )
                    
                    if retrain_success:
                        print("✅ Model retrained successfully!")
                        # Save updated model
                        self.save_system("models/heart_disease_model_updated.pkl")
                    else:
                        print("❌ Model retraining failed!")
                
                # Remove processed records
                self.accumulated_valid_records = self.accumulated_valid_records[self.retrain_batch_size:]
                print(f"📝 Remaining accumulated records: {len(self.accumulated_valid_records)}")
                
            except Exception as e:
                print(f"❌ Error in training/retraining: {e}")
            
            print("=" * 100)
        else:
            model_status = "✅ Model ready" if self.predictor.is_trained else "❌ No model yet"
            print(f"⏳ Need {self.retrain_batch_size} records for {'retraining' if self.predictor.is_trained else 'initial training'}, currently have {len(self.accumulated_valid_records)} ({model_status})")
            print("📋 Current accumulated records:")
            for i, row in enumerate(self.accumulated_valid_records[-5:]):  # Show last 5
                row_dict = row.asDict() if hasattr(row, 'asDict') else row
                print(f"   {i+1}. ID: {row_dict.get('id', 'N/A')}, Age: {row_dict.get('age', 'N/A')}, Num: {row_dict.get('num', 'N/A')}")
    
    def initial_training(self, training_data_path):
        """Initial training with a dataset"""
        print("\n=== INITIAL MODEL TRAINING ===")
        
        # Load and preprocess data
        data = self.preprocessor.load_data(training_data_path)
        if data is None:
            return False
        
        self.preprocessor.explore_data(data)
        
        # Encode and prepare data
        data_encoded = self.preprocessor.encode_categorical_features(data)
        X, y = self.preprocessor.prepare_features_target(data_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Train model
        self.predictor.train_model(X_train_scaled, y_train, tune_hyperparameters=True)
        
        # Evaluate
        if len(X_test_scaled) > 0:
            y_pred = self.predictor.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Initial model accuracy: {accuracy:.4f}")
        
        # Save model
        self.save_system("models/heart_disease_model_initial.pkl")
        
        print("✅ Initial training completed!")
        return True
    
    def start_stream(self, processing_time="5 seconds"):
        """Start Kafka stream processing"""
        if not self.spark:
            raise Exception("Spark session not created. Call create_spark_session() first.")
        
        # Create Kafka stream
        kafka_df = self.create_kafka_stream()
        
        # Parse data
        parsed_df = self.parse_data(kafka_df)
        
        # Start stream
        self.query = parsed_df.writeStream \
            .outputMode("append") \
            .option("checkpointLocation", "checkpoint/heart_disease_kafka/") \
            .trigger(processingTime=processing_time) \
            .foreachBatch(self.process_batch) \
            .start()
        
        print("✓ Kafka stream started")
        print("🔄 Processing: Prediction for num=null, Retraining for valid num")
        return self.query
    
    def wait_for_termination(self):
        """Wait for stream termination"""
        if self.query:
            self.query.awaitTermination()
    
    def stop_stream(self):
        """Stop stream"""
        if self.query:
            self.query.stop()
            print("✓ Stream stopped")
    
    def stop_spark(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            print("✓ Spark session stopped")
    
    def save_system(self, model_path):
        """Save complete system"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            system_data = {
                'model': self.predictor.model,
                'label_encoders': self.preprocessor.label_encoders,
                'scaler': self.preprocessor.scaler,
                'feature_names': self.preprocessor.feature_names,
                'is_trained': self.predictor.is_trained
            }
            
            joblib.dump(system_data, model_path)
            print(f"✅ Complete system saved to {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving system: {e}")
            return False
    
    def load_system(self, model_path):
        """Load complete system"""
        try:
            system_data = joblib.load(model_path)
            
            self.predictor.model = system_data['model']
            self.predictor.is_trained = system_data['is_trained']
            self.preprocessor.label_encoders = system_data['label_encoders']
            self.preprocessor.scaler = system_data['scaler']
            self.preprocessor.feature_names = system_data['feature_names']
            
            print(f"✅ Complete system loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return False
    
    def _create_initial_model_from_stream(self, training_records):
        """Create initial model from streaming data when no model exists"""
        try:
            print("🔧 Creating initial model from streaming data...")
            
            # Convert list of Row objects to DataFrame
            data_dicts = []
            for row in training_records:
                if hasattr(row, 'asDict'):
                    data_dicts.append(row.asDict())
                else:
                    data_dicts.append(row)
            
            training_df = pd.DataFrame(data_dicts)
            print(f"📊 Initial training data shape: {training_df.shape}")
            
            # Validate minimum requirements for training
            if len(training_df) < 3:
                print("❌ Insufficient data for initial training (minimum 3 records required)")
                return False
            
            # Check class distribution
            class_counts = training_df['num'].value_counts()
            print(f"📈 Class distribution: {class_counts.to_dict()}")
            
            if len(class_counts) < 2:
                print("⚠️ Warning: Only one class present in training data. Adding synthetic balance...")
                # Create a balanced dataset by duplicating and modifying samples
                minority_class = 1 - training_df['num'].iloc[0]  # Get the opposite class
                
                # Create synthetic samples by slightly modifying existing ones
                num_synthetic = max(2, len(training_df) // 2)  # Create at least 2 synthetic samples
                synthetic_rows = []
                
                for i in range(num_synthetic):
                    base_row = training_df.iloc[i % len(training_df)].copy()
                    base_row['num'] = minority_class
                    base_row['id'] = base_row['id'] + 10000 + i  # Unique ID
                    
                    # Add small random variations to numeric features (optional)
                    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
                    for col in numeric_cols:
                        if col in base_row.index and pd.notna(base_row[col]):
                            # Add small random variation (±5%)
                            variation = np.random.normal(0, 0.05) * base_row[col]
                            base_row[col] = max(0, base_row[col] + variation)
                    
                    synthetic_rows.append(base_row)
                
                synthetic_df = pd.DataFrame(synthetic_rows)
                training_df = pd.concat([training_df, synthetic_df], ignore_index=True)
                print(f"📊 Balanced data shape: {training_df.shape}")
                print(f"📈 New class distribution: {training_df['num'].value_counts().to_dict()}")
            elif class_counts.min() == 1:
                print("⚠️ Warning: One class has only 1 sample. Adding synthetic samples for balance...")
                # Find the minority class
                minority_class = class_counts.idxmin()
                
                # Create 1-2 additional synthetic samples for the minority class
                minority_samples = training_df[training_df['num'] == minority_class]
                synthetic_rows = []
                
                for i in range(2):  # Add 2 synthetic samples
                    base_row = minority_samples.iloc[0].copy()
                    base_row['id'] = base_row['id'] + 20000 + i  # Unique ID
                    
                    # Add small variations
                    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
                    for col in numeric_cols:
                        if col in base_row.index and pd.notna(base_row[col]):
                            variation = np.random.normal(0, 0.03) * base_row[col]
                            base_row[col] = max(0, base_row[col] + variation)
                    
                    synthetic_rows.append(base_row)
                
                synthetic_df = pd.DataFrame(synthetic_rows)
                training_df = pd.concat([training_df, synthetic_df], ignore_index=True)
                print(f"📊 Augmented data shape: {training_df.shape}")
                print(f"📈 New class distribution: {training_df['num'].value_counts().to_dict()}")
            
            # Basic data exploration
            print("\n=== INITIAL MODEL TRAINING DATA EXPLORATION ===")
            print(f"Features: {[col for col in training_df.columns if col not in ['id', 'num']]}")
            print(f"Target distribution: {training_df['num'].value_counts().to_dict()}")
            
            # Encode categorical features
            training_encoded = self.preprocessor.encode_categorical_features(training_df)
            X, y = self.preprocessor.prepare_features_target(training_encoded)
            
            print(f"✅ Feature encoding completed. Feature count: {len(X.columns)}")
            
            # For initial training with small data, use all data for training
            # and create a small validation split if possible
            if len(X) >= 4:
                # Check if we can use stratify (need at least 2 samples per class)
                class_counts = y.value_counts()
                min_class_count = class_counts.min()
                
                if min_class_count >= 2:
                    # Safe to use stratify
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.25, random_state=42, stratify=y
                    )
                    print(f"📊 Train: {len(X_train)}, Validation: {len(X_val)} (stratified)")
                else:
                    # Cannot use stratify, use random split
                    print(f"⚠️ Minimum class has only {min_class_count} samples, using random split")
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.25, random_state=42, stratify=None
                    )
                    print(f"📊 Train: {len(X_train)}, Validation: {len(X_val)} (random split)")
            else:
                # Use all data for training when very small dataset
                X_train, y_train = X, y
                X_val, y_val = None, None
                print(f"📊 Using all {len(X_train)} records for training (no validation split)")
            
            # Scale features
            if X_val is not None:
                X_train_scaled, X_val_scaled = self.preprocessor.scale_features(X_train, X_val)
            else:
                X_train_scaled = self.preprocessor.scale_features(X_train)
                X_val_scaled = None
            
            # Train initial model (no hyperparameter tuning for small initial dataset)
            print("🎯 Training initial Random Forest model...")
            self.predictor.train_model(X_train_scaled, y_train, tune_hyperparameters=False)
            
            # Quick validation if validation set exists
            if X_val_scaled is not None and len(X_val_scaled) > 0:
                y_val_pred = self.predictor.model.predict(X_val_scaled)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                print(f"🎯 Initial validation accuracy: {val_accuracy:.4f}")
            else:
                # Use training accuracy as a baseline
                y_train_pred = self.predictor.model.predict(X_train_scaled)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"🎯 Initial training accuracy: {train_accuracy:.4f}")
            
            print("🎉 Initial model creation completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating initial model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Cleanup all resources"""
        self.stop_stream()
        self.stop_spark()
        print("✓ Resources cleaned up")

def main():
    """Main function"""
    system = None
    
    try:
        print("🚀 Starting Heart Disease Kafka ML System...")
        
        # Create system instance
        system = HeartDiseaseKafkaSystem(
            topic_name='BTLCDDDHeart',
            kafka_server='kafka:9092',
            model_path='models/heart_disease_model_initial.pkl'
        )
        
        # Initialize Spark session
        system.create_spark_session()
        
        # Check if model is trained, if not, wait for initial training data
        if not system.predictor.is_trained:
            print("\n⚠️ No trained model found.")
            print("🔄 System will automatically create initial model when receiving first 5 valid records (num≠null)")
            print("💡 You can also provide initial training data manually:")
            print("   system.initial_training('path/to/training_data.csv')")
            
            # Example: Uncomment the following line and provide your training data path
            # system.initial_training('data/heart_disease_train.csv')
        
        # Start stream processing
        system.start_stream(processing_time="10 seconds")
        
        print("\n📊 System Status:")
        print(f"   🤖 Model trained: {system.predictor.is_trained}")
        print(f"   📚 Retrain batch size: {system.retrain_batch_size}")
        print(f"   🔄 Processing logic:")
        print(f"      • num=null → Prediction")
        print(f"      • num≠null → Accumulate for retraining")
        print("\n💡 Press Ctrl+C to stop...")
        print("-" * 60)
        
        # Wait for data
        system.wait_for_termination()
        
    except KeyboardInterrupt:
        print("\n⏹ Stopping system...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if system:
            system.cleanup()

if __name__ == "__main__":
    main()