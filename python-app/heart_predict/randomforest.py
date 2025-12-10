import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, col, lit, struct
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
    SPARK_AVAILABLE = True
except ImportError:
    print("Warning: PySpark not available. Falling back to pandas-only implementation.")
    SPARK_AVAILABLE = False

class DataPreprocessor:
    """Class for handling data preprocessing operations"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def explore_data(self, data):
        """Perform basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print("\nDataset Info:")
        print(data.info())
        
        print("\nDataset Description:")
        print(data.describe())
        
        print("\nMissing Values:")
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found")
        
        print("\nTarget Distribution (num):")
        print(data['num'].value_counts().sort_index())
        
        print("\nDataset Distribution (if exists):")
        if 'dataset' in data.columns:
            print(data['dataset'].value_counts())
        
        return data
    
    def encode_categorical_features(self, data):
        """Encode categorical features to numerical values"""
        # Only encode actual categorical columns, skip 'id' and 'dataset'
        categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        
        data_encoded = data.copy()
        
        # Drop non-predictive columns if they exist
        columns_to_drop = []
        if 'id' in data_encoded.columns:
            columns_to_drop.append('id')
        if 'dataset' in data_encoded.columns:
            columns_to_drop.append('dataset')
        
        if columns_to_drop:
            print(f"Dropping non-predictive columns: {columns_to_drop}")
            data_encoded = data_encoded.drop(columns_to_drop, axis=1)
        
        for column in categorical_columns:
            if column in data_encoded.columns:
                # Handle missing values by converting to string first
                data_encoded[column] = data_encoded[column].fillna('unknown').astype(str)
                
                le = LabelEncoder()
                data_encoded[column] = le.fit_transform(data_encoded[column])
                self.label_encoders[column] = le
                print(f"Encoded {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return data_encoded
    
    def prepare_features_target(self, data):
        """Separate features and target variable"""
        # Convert target to binary (0 = no disease, 1+ = disease)
        y = (data['num'] > 0).astype(int)
        
        # Remove non-relevant columns for prediction
        columns_to_drop = ['num', 'id', 'dataset']  # Remove id and dataset columns
        columns_to_drop = [col for col in columns_to_drop if col in data.columns]
        
        X = data.drop(columns_to_drop, axis=1)
        self.feature_names = X.columns.tolist()
        
        print(f"\nFeatures: {self.feature_names}")
        print(f"Target classes: {sorted(y.unique())} (0=No Disease, 1=Disease)")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled

class HeartDiseasePredictor:
    """Main class for heart disease prediction using Random Forest with Spark UDF support"""
    
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        self.is_trained = False
        self.best_params = None
        self.spark = None
        
        # Initialize Spark session if available
        if SPARK_AVAILABLE:
            try:
                self.spark = SparkSession.builder \
                    .appName("HeartDiseasePredictor") \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .getOrCreate()
                print("Spark session initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Spark session: {e}")
                self.spark = None
        
    def train_model(self, X_train, y_train, tune_hyperparameters=True):
        """Train Random Forest model with optional hyperparameter tuning"""
        print("\n=== TRAINING RANDOM FOREST MODEL ===")
        
        # Check if we have enough samples for cross-validation
        n_samples = len(X_train)
        cv_folds = min(5, n_samples)  # Use fewer folds if we have few samples
        
        if n_samples < 5:
            print(f"Warning: Only {n_samples} samples available. Using {cv_folds}-fold CV.")
            tune_hyperparameters = False  # Skip hyperparameter tuning for small datasets
        
        if tune_hyperparameters and n_samples >= 10:  # Need at least 10 samples for meaningful tuning
            print("Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train, y_train)
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Cross-validation score (only if we have enough samples)
        if n_samples >= cv_folds:
            try:
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Cross-validation failed: {e}")
                print("Skipping cross-validation due to insufficient data")
        else:
            print("Skipping cross-validation due to insufficient samples")
        
        return self.model
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        n_samples = len(X_train)
        
        # Adjust CV folds based on sample size
        cv_folds = min(5, max(2, n_samples // 3))  # At least 2 folds, max 5 folds
        
        if n_samples < 10:
            print(f"Warning: Only {n_samples} samples. Using simplified parameter grid.")
            # Use simpler parameter grid for small datasets
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt']
            }
        else:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        try:
            grid_search = GridSearchCV(
                rf, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            print("Using default parameters instead")
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return None
        
        print("\n=== MODEL EVALUATION ===")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def get_feature_importance(self, feature_names):
        """Get and plot feature importance"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return None
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n=== FEATURE IMPORTANCE ===")
        print(feature_importance_df)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def _create_prediction_udf(self, preprocessor):
        """Create pandas UDF for batch predictions"""
        if not SPARK_AVAILABLE or self.spark is None:
            print("Spark not available, using pandas implementation")
            return None
        
        # Define return schema for UDF
        return_schema = StructType([
            StructField("prediction", IntegerType(), True),
            StructField("probability_no_disease", DoubleType(), True),
            StructField("probability_disease", DoubleType(), True),
            StructField("risk_level", StringType(), True)
        ])
        
        # Capture model and preprocessor in closure
        model = self.model
        label_encoders = preprocessor.label_encoders
        scaler = preprocessor.scaler
        feature_names = preprocessor.feature_names
        
        # Fix: Use correct function type for pandas UDF
        try:
            from pyspark.sql.functions import PandasUDFType
            function_type = PandasUDFType.GROUPED_MAP
        except ImportError:
            # For newer versions of PySpark
            function_type = "grouped_map"
        
        @pandas_udf(returnType=return_schema, functionType=function_type)
        def predict_batch_udf(pdf):
            """Pandas UDF for batch predictions"""
            try:
                # Preprocess the batch
                pdf_processed = pdf.copy()
                
                # Encode categorical features
                for column, encoder in label_encoders.items():
                    if column in pdf_processed.columns:
                        try:
                            pdf_processed[column] = pdf_processed[column].fillna('unknown').astype(str)
                            pdf_processed[column] = encoder.transform(pdf_processed[column])
                        except ValueError:
                            # Handle unknown categories
                            pdf_processed[column] = encoder.transform([encoder.classes_[0]] * len(pdf_processed))
                
                # Ensure all required features are present
                for feature in feature_names:
                    if feature not in pdf_processed.columns:
                        pdf_processed[feature] = 0  # Default value for missing features
                
                # Select and order features correctly
                X_batch = pdf_processed[feature_names]
                
                # Scale features
                X_batch_scaled = scaler.transform(X_batch)
                
                # Make predictions
                predictions = model.predict(X_batch_scaled)
                probabilities = model.predict_proba(X_batch_scaled)
                
                # Prepare results
                results = pd.DataFrame({
                    'prediction': predictions.astype(int),
                    'probability_no_disease': probabilities[:, 0],
                    'probability_disease': probabilities[:, 1],
                    'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                                  for p in probabilities[:, 1]]
                })
                
                return results
                
            except Exception as e:
                print(f"Error in prediction UDF: {e}")
                # Return empty DataFrame with correct schema
                return pd.DataFrame({
                    'prediction': [],
                    'probability_no_disease': [],
                    'probability_disease': [],
                    'risk_level': []
                })
        
        return predict_batch_udf
    
    def predict_batch_spark(self, spark_df, preprocessor):
        """Predict using Spark DataFrame with pandas UDF"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return None
        
        if not SPARK_AVAILABLE or self.spark is None:
            print("Spark not available, falling back to pandas")
            return self.predict_batch_pandas(spark_df.toPandas(), preprocessor)
        
        try:
            # Create prediction UDF
            predict_udf = self._create_prediction_udf(preprocessor)
            if predict_udf is None:
                return None
            
            # Apply UDF to get predictions
            # Group by a constant to process all data in one partition
            spark_df_with_key = spark_df.withColumn("group_key", lit(1))
            
            predictions_df = spark_df_with_key.groupBy("group_key").apply(predict_udf)
            
            print("Batch predictions completed using Spark")
            return predictions_df
            
        except Exception as e:
            print(f"Error in Spark batch prediction: {e}")
            print("Falling back to pandas implementation")
            return self.predict_batch_pandas(spark_df.toPandas(), preprocessor)
    
    def predict_batch_pandas(self, df, preprocessor):
        """Predict using pandas DataFrame (fallback method)"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return None
        
        try:
            # Preprocess data
            df_processed = df.copy()
            
            # Encode categorical features
            for column, encoder in preprocessor.label_encoders.items():
                if column in df_processed.columns:
                    try:
                        df_processed[column] = df_processed[column].fillna('unknown').astype(str)
                        df_processed[column] = encoder.transform(df_processed[column])
                    except ValueError:
                        df_processed[column] = encoder.transform([encoder.classes_[0]] * len(df_processed))
            
            # Ensure all required features are present
            for feature in preprocessor.feature_names:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0
            
            # Select and order features
            X_batch = df_processed[preprocessor.feature_names]
            
            # Scale features
            X_batch_scaled = preprocessor.scaler.transform(X_batch)
            
            # Make predictions
            predictions = self.model.predict(X_batch_scaled)
            probabilities = self.model.predict_proba(X_batch_scaled)
            
            # Prepare results
            results = pd.DataFrame({
                'prediction': predictions.astype(int),
                'probability_no_disease': probabilities[:, 0],
                'probability_disease': probabilities[:, 1],
                'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                              for p in probabilities[:, 1]]
            })
            
            print("Batch predictions completed using pandas")
            return results
            
        except Exception as e:
            print(f"Error in pandas batch prediction: {e}")
            return None
    
    def predict_single_patient(self, patient_data, preprocessor):
        """Predict heart disease for a single patient"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return None
        
        # Convert to DataFrame if it's a dict
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Use batch prediction for single patient
        result_df = self.predict_batch_pandas(patient_df, preprocessor)
        
        if result_df is not None and len(result_df) > 0:
            result = result_df.iloc[0].to_dict()
            return result
        
        return None
    
    def retrain_with_multiple_datasets(self, dataset_paths, preprocessor, retrain_type='incremental', tune_hyperparameters=False):
        """
        Retrain model with multiple datasets using pandas UDF for processing
        
        Parameters:
        - dataset_paths: list of paths to CSV files or list of DataFrames
        - preprocessor: DataPreprocessor instance
        - retrain_type: 'retrain' (only new data) or 'incremental' (combine with existing)
        - tune_hyperparameters: whether to tune hyperparameters
        """
        print(f"\n=== RETRAINING WITH {len(dataset_paths)} DATASETS ({retrain_type.upper()}) ===")
        
        all_X_new = []
        all_y_new = []
        
        # Process each dataset
        for i, dataset_path in enumerate(dataset_paths):
            print(f"\nProcessing dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
            
            try:
                # Load data
                if isinstance(dataset_path, str):
                    new_data = preprocessor.load_data(dataset_path)
                elif isinstance(dataset_path, pd.DataFrame):
                    new_data = dataset_path.copy()
                else:
                    print(f"Unsupported data type for dataset {i+1}")
                    continue
                
                if new_data is None:
                    print(f"Failed to load dataset {i+1}")
                    continue
                
                # Preprocess new data using existing encoders
                new_data_encoded = self._preprocess_new_data(new_data, preprocessor)
                X_new, y_new = preprocessor.prepare_features_target(new_data_encoded)
                
                # Scale new features
                X_new_scaled = pd.DataFrame(
                    preprocessor.scaler.transform(X_new),
                    columns=preprocessor.feature_names
                )
                
                all_X_new.append(X_new_scaled)
                all_y_new.append(pd.Series(y_new))
                
                print(f"Dataset {i+1} processed successfully. Shape: {X_new_scaled.shape}")
                
            except Exception as e:
                print(f"Error processing dataset {i+1}: {e}")
                continue
        
        if not all_X_new:
            print("No datasets were successfully processed")
            return False
        
        # Combine all new data
        print("\nCombining all new datasets...")
        X_combined_new = pd.concat(all_X_new, axis=0, ignore_index=True)
        y_combined_new = pd.concat(all_y_new, axis=0, ignore_index=True)
        
        print(f"Combined new data shape: {X_combined_new.shape}")
        print(f"Combined target distribution: {y_combined_new.value_counts().to_dict()}")
        
        # Create retrain UDF if Spark is available
        if SPARK_AVAILABLE and self.spark is not None:
            try:
                return self._retrain_with_spark_udf(X_combined_new, y_combined_new, retrain_type, tune_hyperparameters)
            except Exception as e:
                print(f"Spark retraining failed: {e}")
                print("Falling back to pandas implementation")
        
        # Fallback to pandas implementation
        return self._retrain_with_pandas(X_combined_new, y_combined_new, retrain_type, tune_hyperparameters)
    
    def _retrain_with_spark_udf(self, X_new, y_new, retrain_type, tune_hyperparameters):
        """Retrain model using Spark UDF for data processing"""
        print("Using Spark UDF for retraining...")
        
        try:
            # Convert to Spark DataFrame
            combined_data = pd.concat([X_new, y_new.rename('target')], axis=1)
            spark_df = self.spark.createDataFrame(combined_data)
            
            # Define return schema for UDF
            return_schema = StructType([
                StructField("status", StringType(), True),
                StructField("num_samples", IntegerType(), True),
                StructField("num_features", IntegerType(), True)
            ])
            
            # Fix: Use correct function type for pandas UDF
            try:
                from pyspark.sql.functions import PandasUDFType
                function_type = PandasUDFType.GROUPED_MAP
            except ImportError:
                # For newer versions of PySpark
                function_type = "grouped_map"
            
            @pandas_udf(returnType=return_schema, functionType=function_type)
            def prepare_retrain_data_udf(pdf):
                """UDF to prepare data for retraining"""
                try:
                    # Separate features and target
                    X = pdf.drop('target', axis=1)
                    y = pdf['target']
                    
                    # Validate data
                    if len(X) == 0:
                        return pd.DataFrame({'status': ['error: no data'], 'num_samples': [0], 'num_features': [0]})
                    
                    # Check for class balance
                    class_counts = y.value_counts()
                    if len(class_counts) < 2:
                        return pd.DataFrame({'status': ['warning: single class'], 'num_samples': [len(X)], 'num_features': [len(X.columns)]})
                    
                    return pd.DataFrame({'status': ['data_prepared'], 'num_samples': [len(X)], 'num_features': [len(X.columns)]})
                except Exception as e:
                    return pd.DataFrame({'status': [f'error: {e}'], 'num_samples': [0], 'num_features': [0]})
            
            # Apply UDF (mainly for validation)
            spark_df_with_key = spark_df.withColumn("group_key", lit(1))
            result = spark_df_with_key.groupBy("group_key").apply(prepare_retrain_data_udf).collect()
            
            if result and len(result) > 0:
                status = result[0]['status']
                num_samples = result[0]['num_samples']
                num_features = result[0]['num_features']
                
                print(f"Spark UDF validation: {status}, Samples: {num_samples}, Features: {num_features}")
                
                if status == 'data_prepared':
                    print("Data preparation completed with Spark UDF")
                    return self._retrain_with_pandas(X_new, y_new, retrain_type, tune_hyperparameters)
                else:
                    print(f"Data validation issue: {status}")
                    return False
            else:
                print("No result from Spark UDF")
                return False
                
        except Exception as e:
            print(f"Spark UDF processing failed: {e}")
            print("Falling back to pandas implementation")
            return self._retrain_with_pandas(X_new, y_new, retrain_type, tune_hyperparameters)
    
    def _retrain_with_pandas(self, X_new, y_new, retrain_type, tune_hyperparameters):
        """Retrain model using pandas (actual training logic)"""
        if retrain_type == 'retrain':
            # Retrain only on new data
            print("Retraining on new data only...")
            self.train_model(X_new, y_new, tune_hyperparameters)
            
        elif retrain_type == 'incremental':
            # This would need access to original training data
            # For now, just retrain on new data
            print("Incremental training not fully implemented, retraining on new data...")
            self.train_model(X_new, y_new, tune_hyperparameters)
        
        print("Retraining completed!")
        return True
    
    def _preprocess_new_data(self, new_data, preprocessor):
        """Preprocess new data using existing encoders"""
        data_encoded = new_data.copy()
        
        # Drop non-predictive columns if they exist
        columns_to_drop = []
        if 'id' in data_encoded.columns:
            columns_to_drop.append('id')
        if 'dataset' in data_encoded.columns:
            columns_to_drop.append('dataset')
        
        if columns_to_drop:
            data_encoded = data_encoded.drop(columns_to_drop, axis=1)
        
        # Encode categorical features using existing encoders
        for column, encoder in preprocessor.label_encoders.items():
            if column in data_encoded.columns:
                # Handle missing values
                data_encoded[column] = data_encoded[column].fillna('unknown').astype(str)
                
                # Handle unknown categories
                try:
                    data_encoded[column] = encoder.transform(data_encoded[column])
                except ValueError as e:
                    print(f"Warning: Unknown categories in {column}. Using most frequent class.")
                    # Replace unknown categories with the most frequent class
                    known_categories = encoder.classes_
                    data_encoded[column] = data_encoded[column].apply(
                        lambda x: x if x in known_categories else known_categories[0]
                    )
                    data_encoded[column] = encoder.transform(data_encoded[column])
        
        return data_encoded
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            print("Error: Model is not trained yet!")
            return False
        
        try:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def __del__(self):
        """Cleanup Spark session"""
        if hasattr(self, 'spark') and self.spark is not None:
            try:
                self.spark.stop()
            except:
                pass

class HeartDiseaseAnalysisSystem:
    """Main system class that orchestrates the entire process with Spark support"""
    
    def __init__(self, random_state=42):
        self.preprocessor = DataPreprocessor()
        self.predictor = HeartDiseasePredictor(random_state)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
    
    def run_complete_analysis(self, csv_file_path, test_size=0.2, tune_hyperparameters=True):
        """Run complete analysis pipeline"""
        print("Starting Heart Disease Prediction Analysis...")
        
        # Load and explore data
        self.data = self.preprocessor.load_data(csv_file_path)
        if self.data is None:
            return None
        
        self.preprocessor.explore_data(self.data)
        
        # Preprocess data
        data_encoded = self.preprocessor.encode_categorical_features(self.data)
        X, y = self.preprocessor.prepare_features_target(data_encoded)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Scale features
        self.X_train_scaled, self.X_test_scaled = self.preprocessor.scale_features(
            self.X_train, self.X_test
        )
        
        # Train model
        self.predictor.train_model(self.X_train_scaled, self.y_train, tune_hyperparameters)
        
        # Evaluate model
        evaluation_results = self.predictor.evaluate_model(self.X_test_scaled, self.y_test)
        
        # Feature importance
        self.predictor.get_feature_importance(self.preprocessor.feature_names)
        
        return evaluation_results
    
    def predict_batch(self, data, use_spark=True):
        """Predict for batch of patients using Spark UDF or pandas"""
        if use_spark and SPARK_AVAILABLE and self.predictor.spark is not None:
            # Convert to Spark DataFrame if needed
            if isinstance(data, pd.DataFrame):
                spark_df = self.predictor.spark.createDataFrame(data)
            else:
                spark_df = data
            
            return self.predictor.predict_batch_spark(spark_df, self.preprocessor)
        else:
            # Use pandas implementation
            if not isinstance(data, pd.DataFrame):
                data = data.toPandas()  # Convert from Spark to pandas
            
            return self.predictor.predict_batch_pandas(data, self.preprocessor)
    
    def retrain_with_multiple_files(self, file_paths, retrain_type='incremental', tune_hyperparameters=False):
        """Retrain model with multiple CSV files"""
        return self.predictor.retrain_with_multiple_datasets(
            file_paths, self.preprocessor, retrain_type, tune_hyperparameters
        )
    
    def retrain_with_dataframes(self, dataframes, retrain_type='incremental', tune_hyperparameters=False):
        """Retrain model with multiple DataFrames"""
        return self.predictor.retrain_with_multiple_datasets(
            dataframes, self.preprocessor, retrain_type, tune_hyperparameters
        )
    
    def load_pretrained_model(self, model_path, preprocessor_path=None):
        """Load a pre-trained model and optionally its preprocessor"""
        success = self.predictor.load_model(model_path)
        
        if preprocessor_path and success:
            try:
                # Load preprocessor components
                preprocessor_data = joblib.load(preprocessor_path)
                self.preprocessor.label_encoders = preprocessor_data['label_encoders']
                self.preprocessor.scaler = preprocessor_data['scaler']
                self.preprocessor.feature_names = preprocessor_data['feature_names']
                print(f"Preprocessor loaded from {preprocessor_path}")
            except Exception as e:
                print(f"Warning: Could not load preprocessor: {str(e)}")
                print("You may need to retrain the preprocessor with new data")
        
        return success
    
    def save_complete_system(self, model_path, preprocessor_path):
        """Save both model and preprocessor"""
        # Save model
        model_saved = self.predictor.save_model(model_path)
        
        # Save preprocessor components
        preprocessor_saved = False
        try:
            preprocessor_data = {
                'label_encoders': self.preprocessor.label_encoders,
                'scaler': self.preprocessor.scaler,
                'feature_names': self.preprocessor.feature_names
            }
            joblib.dump(preprocessor_data, preprocessor_path)
            print(f"Preprocessor saved to {preprocessor_path}")
            preprocessor_saved = True
        except Exception as e:
            print(f"Error saving preprocessor: {str(e)}")
        
        return model_saved and preprocessor_saved
    
    def predict_new_patient(self, patient_data):
        """Predict for a new patient"""
        result = self.predictor.predict_single_patient(patient_data, self.preprocessor)
        
        if result:
            print(f"\n=== PREDICTION RESULT ===")
            print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
            print(f"Probability of Disease: {result['probability_disease']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
        
        return result

def main():
    """Main function to run the heart disease prediction system"""
    
    # Initialize the system
    system = HeartDiseaseAnalysisSystem(random_state=42)
    
    # Path to your CSV file
    csv_file_path = "sample_heart_data.csv"  # Replace with your actual file path
    
    try:
        # Choice 1: Train new model
        print("="*60)
        print("OPTION 1: TRAINING NEW MODEL")
        print("="*60)
        
        # Run complete analysis
        results = system.run_complete_analysis(
            csv_file_path, 
            test_size=0.2, 
            tune_hyperparameters=True  # Set to False for faster training
        )
        
        if results is None:
            print("Analysis failed. Please check your data file.")
            return
        
        # Save the complete system (model + preprocessor)
        system.save_complete_system(
            "heart_disease_rf_model.pkl", 
            "heart_disease_preprocessor.pkl"
        )
        
        # Example: Predict for a single patient
        new_patient = {
            'age': 65,
            'sex': 'Male',
            'cp': 'typical angina',
            'trestbps': 140,
            'chol': 250,
            'fbs': 'True',
            'restecg': 'normal',
            'thalch': 150,
            'exang': 'True',
            'oldpeak': 2.5,
            'slope': 'upsloping',
            'ca': 1,
            'thal': 'fixed defect'
        }
        
        print("\n" + "="*50)
        print("TESTING WITH NEW PATIENT DATA")
        print("="*50)
        
        prediction_result = system.predict_new_patient(new_patient)
        
        # Example: Batch predictions using Spark UDF
        print("\n" + "="*50)
        print("TESTING BATCH PREDICTIONS WITH SPARK UDF")
        print("="*50)
        
        # Create sample batch data
        batch_patients = pd.DataFrame([
            {
                'age': 65, 'sex': 'Male', 'cp': 'typical angina', 'trestbps': 140,
                'chol': 250, 'fbs': 'True', 'restecg': 'normal', 'thalch': 150,
                'exang': 'True', 'oldpeak': 2.5, 'slope': 'upsloping', 'ca': 1, 'thal': 'fixed defect'
            },
            {
                'age': 45, 'sex': 'Female', 'cp': 'atypical angina', 'trestbps': 120,
                'chol': 200, 'fbs': 'False', 'restecg': 'normal', 'thalch': 170,
                'exang': 'False', 'oldpeak': 1.0, 'slope': 'flat', 'ca': 0, 'thal': 'normal'
            },
            {
                'age': 55, 'sex': 'Male', 'cp': 'non-anginal pain', 'trestbps': 130,
                'chol': 220, 'fbs': 'False', 'restecg': 'abnormal', 'thalch': 160,
                'exang': 'True', 'oldpeak': 2.0, 'slope': 'downsloping', 'ca': 2, 'thal': 'reversible defect'
            }
        ])
        
        # Test batch predictions
        batch_results = system.predict_batch(batch_patients, use_spark=True)
        if batch_results is not None:
            if hasattr(batch_results, 'show'):  # Spark DataFrame
                print("Spark UDF Batch Prediction Results:")
                batch_results.show()
            else:  # Pandas DataFrame
                print("Pandas Batch Prediction Results:")
                print(batch_results)
        
        # Choice 2: Demonstrate retraining with multiple datasets
        print("\n" + "="*60)
        print("OPTION 2: RETRAINING WITH MULTIPLE DATASETS")
        print("="*60)
        
        # Example with multiple dataset paths
        # Create more realistic sample data for testing
        print("Creating additional sample datasets for testing...")
        
        # Sample data 1 - High risk patients
        sample_data_1 = pd.DataFrame([
            {
                'age': 70, 'sex': 'Male', 'cp': 'typical angina', 'trestbps': 150,
                'chol': 280, 'fbs': 'True', 'restecg': 'abnormal', 'thalch': 140,
                'exang': 'True', 'oldpeak': 3.0, 'slope': 'downsloping', 'ca': 2, 
                'thal': 'reversible defect', 'num': 1
            },
            {
                'age': 65, 'sex': 'Male', 'cp': 'atypical angina', 'trestbps': 145,
                'chol': 260, 'fbs': 'True', 'restecg': 'abnormal', 'thalch': 135,
                'exang': 'True', 'oldpeak': 2.8, 'slope': 'flat', 'ca': 1, 
                'thal': 'fixed defect', 'num': 1
            },
            {
                'age': 68, 'sex': 'Female', 'cp': 'non-anginal pain', 'trestbps': 155,
                'chol': 290, 'fbs': 'False', 'restecg': 'abnormal', 'thalch': 130,
                'exang': 'True', 'oldpeak': 3.2, 'slope': 'downsloping', 'ca': 2, 
                'thal': 'reversible defect', 'num': 1
            },
            {
                'age': 72, 'sex': 'Male', 'cp': 'typical angina', 'trestbps': 160,
                'chol': 300, 'fbs': 'True', 'restecg': 'abnormal', 'thalch': 125,
                'exang': 'True', 'oldpeak': 3.5, 'slope': 'downsloping', 'ca': 3, 
                'thal': 'fixed defect', 'num': 2
            },
            {
                'age': 75, 'sex': 'Male', 'cp': 'asymptomatic', 'trestbps': 165,
                'chol': 310, 'fbs': 'True', 'restecg': 'abnormal', 'thalch': 120,
                'exang': 'True', 'oldpeak': 4.0, 'slope': 'downsloping', 'ca': 3, 
                'thal': 'fixed defect', 'num': 3
            }
        ])
        
        # Sample data 2 - Low risk patients
        sample_data_2 = pd.DataFrame([
            {
                'age': 35, 'sex': 'Female', 'cp': 'atypical angina', 'trestbps': 110,
                'chol': 180, 'fbs': 'False', 'restecg': 'normal', 'thalch': 180,
                'exang': 'False', 'oldpeak': 0.5, 'slope': 'upsloping', 'ca': 0, 
                'thal': 'normal', 'num': 0
            },
            {
                'age': 40, 'sex': 'Male', 'cp': 'non-anginal pain', 'trestbps': 115,
                'chol': 190, 'fbs': 'False', 'restecg': 'normal', 'thalch': 175,
                'exang': 'False', 'oldpeak': 0.8, 'slope': 'upsloping', 'ca': 0, 
                'thal': 'normal', 'num': 0
            },
            {
                'age': 45, 'sex': 'Female', 'cp': 'asymptomatic', 'trestbps': 120,
                'chol': 200, 'fbs': 'False', 'restecg': 'normal', 'thalch': 170,
                'exang': 'False', 'oldpeak': 1.0, 'slope': 'flat', 'ca': 0, 
                'thal': 'normal', 'num': 0
            },
            {
                'age': 38, 'sex': 'Male', 'cp': 'typical angina', 'trestbps': 118,
                'chol': 185, 'fbs': 'False', 'restecg': 'normal', 'thalch': 185,
                'exang': 'False', 'oldpeak': 0.3, 'slope': 'upsloping', 'ca': 0, 
                'thal': 'normal', 'num': 0
            },
            {
                'age': 42, 'sex': 'Female', 'cp': 'non-anginal pain', 'trestbps': 112,
                'chol': 195, 'fbs': 'False', 'restecg': 'normal', 'thalch': 178,
                'exang': 'False', 'oldpeak': 0.6, 'slope': 'upsloping', 'ca': 0, 
                'thal': 'normal', 'num': 0
            }
        ])
        
        # Mixed data
        sample_data_3 = pd.DataFrame([
            {
                'age': 55, 'sex': 'Male', 'cp': 'atypical angina', 'trestbps': 130,
                'chol': 220, 'fbs': 'False', 'restecg': 'normal', 'thalch': 160,
                'exang': 'False', 'oldpeak': 1.5, 'slope': 'flat', 'ca': 1, 
                'thal': 'normal', 'num': 0
            },
            {
                'age': 60, 'sex': 'Male', 'cp': 'non-anginal pain', 'trestbps': 135,
                'chol': 240, 'fbs': 'False', 'restecg': 'normal', 'thalch': 155,
                'exang': 'True', 'oldpeak': 1.8, 'slope': 'flat', 'ca': 1, 
                'thal': 'reversible defect', 'num': 1
            },
            {
                'age': 50, 'sex': 'Female', 'cp': 'asymptomatic', 'trestbps': 125,
                'chol': 210, 'fbs': 'False', 'restecg': 'abnormal', 'thalch': 165,
                'exang': 'False', 'oldpeak': 1.2, 'slope': 'upsloping', 'ca': 0, 
                'thal': 'normal', 'num': 0
            },
            {
                'age': 58, 'sex': 'Male', 'cp': 'typical angina', 'trestbps': 140,
                'chol': 250, 'fbs': 'True', 'restecg': 'abnormal', 'thalch': 150,
                'exang': 'True', 'oldpeak': 2.2, 'slope': 'downsloping', 'ca': 2, 
                'thal': 'fixed defect', 'num': 1
            }
        ])
        
        multiple_dataframes = [sample_data_1, sample_data_2, sample_data_3]
        
        print(f"Created {len(multiple_dataframes)} sample datasets with sizes: {[len(df) for df in multiple_dataframes]}")
        
        # Test retraining with multiple DataFrames
        print("Retraining with multiple DataFrames...")
        retrain_success = system.retrain_with_dataframes(
            multiple_dataframes,
            retrain_type='incremental',
            tune_hyperparameters=False  # Disable for small datasets
        )
        
        if retrain_success:
            print("Retraining with multiple datasets completed successfully!")
            
            # Test predictions after retraining
            retrained_prediction = system.predict_new_patient(new_patient)
            
            print(f"\nComparison:")
            print(f"Original prediction: {prediction_result['prediction'] if prediction_result else 'N/A'}")
            print(f"Retrained prediction: {retrained_prediction['prediction'] if retrained_prediction else 'N/A'}")
        
        # Choice 3: Load and use pretrained model
        print("\n" + "="*60)
        print("OPTION 3: LOADING PRETRAINED MODEL")
        print("="*60)
        
        # Create new system instance to simulate loading
        new_system = HeartDiseaseAnalysisSystem(random_state=42)
        
        # Load pretrained model and preprocessor
        if new_system.load_pretrained_model(
            "heart_disease_rf_model.pkl", 
            "heart_disease_preprocessor.pkl"
        ):
            print("Successfully loaded pretrained model!")
            
            # Test prediction with loaded model
            loaded_prediction = new_system.predict_new_patient(new_patient)
            
            # Test batch prediction with loaded model
            loaded_batch_results = new_system.predict_batch(batch_patients, use_spark=True)
            
            print("Loaded model predictions completed!")
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file '{csv_file_path}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

# Additional utility functions for advanced usage with Spark support
def load_and_predict_batch_spark(model_path, preprocessor_path, spark_df):
    """Utility function to load model and make batch predictions using Spark"""
    system = HeartDiseaseAnalysisSystem()
    
    if system.load_pretrained_model(model_path, preprocessor_path):
        return system.predict_batch(spark_df, use_spark=True)
    else:
        print("Failed to load model")
        return None

def retrain_with_multiple_sources(model_path, preprocessor_path, data_sources, retrain_type='incremental'):
    """
    Utility function to retrain existing model with multiple data sources
    
    Parameters:
    - model_path: path to existing model
    - preprocessor_path: path to existing preprocessor
    - data_sources: list of file paths or DataFrames
    - retrain_type: 'retrain' or 'incremental'
    """
    system = HeartDiseaseAnalysisSystem()
    
    if system.load_pretrained_model(model_path, preprocessor_path):
        success = system.retrain_with_multiple_files(
            data_sources, 
            retrain_type=retrain_type,
            tune_hyperparameters=False
        )
        
        if success:
            # Save retrained model
            system.save_complete_system(
                model_path.replace('.pkl', '_retrained.pkl'),
                preprocessor_path.replace('.pkl', '_retrained.pkl')
            )
            print("Retrained model saved!")
        
        return success
    else:
        print("Failed to load existing model")
        return False

def create_spark_session_with_config():
    """Create optimized Spark session for heart disease prediction"""
    if not SPARK_AVAILABLE:
        print("PySpark not available")
        return None
    
    try:
        spark = SparkSession.builder \
            .appName("HeartDiseasePredictor") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        print("Optimized Spark session created successfully")
        return spark
    except Exception as e:
        print(f"Failed to create Spark session: {e}")
        return None

# Example usage patterns for different scenarios
def example_usage_patterns():
    """Examples of different usage patterns"""
    
    print("="*60)
    print("EXAMPLE USAGE PATTERNS")
    print("="*60)
    
    # Pattern 1: Single dataset training
    print("\n1. Single dataset training:")
    print("""
    system = HeartDiseaseAnalysisSystem()
    results = system.run_complete_analysis('heart_data.csv')
    system.save_complete_system('model.pkl', 'preprocessor.pkl')
    """)
    
    # Pattern 2: Batch predictions with Spark
    print("\n2. Batch predictions with Spark UDF:")
    print("""
    system = HeartDiseaseAnalysisSystem()
    system.load_pretrained_model('model.pkl', 'preprocessor.pkl')
    
    # With pandas DataFrame
    batch_results = system.predict_batch(patient_df, use_spark=True)
    
    # With Spark DataFrame
    spark_df = spark.createDataFrame(patient_df)
    batch_results = system.predict_batch(spark_df, use_spark=True)
    """)
    
    # Pattern 3: Multi-dataset retraining
    print("\n3. Multi-dataset retraining:")
    print("""
    # With file paths
    file_paths = ['data1.csv', 'data2.csv', 'data3.csv']
    system.retrain_with_multiple_files(file_paths, retrain_type='incremental')
    
    # With DataFrames
    dataframes = [df1, df2, df3]
    system.retrain_with_dataframes(dataframes, retrain_type='incremental')
    """)
    
    # Pattern 4: Production pipeline
    print("\n4. Production pipeline:")
    print("""
    # Load model once
    system = HeartDiseaseAnalysisSystem()
    system.load_pretrained_model('production_model.pkl', 'production_preprocessor.pkl')
    
    # Process streaming data
    for batch_df in streaming_data:
        predictions = system.predict_batch(batch_df, use_spark=True)
        # Process predictions...
    
    # Periodic retraining
    new_data_files = collect_new_training_data()
    system.retrain_with_multiple_files(new_data_files, retrain_type='incremental')
    system.save_complete_system('updated_model.pkl', 'updated_preprocessor.pkl')
    """)

if __name__ == "__main__":
    # Run main analysis
    main()
    
    # Show example usage patterns
    # example_usage_patterns()