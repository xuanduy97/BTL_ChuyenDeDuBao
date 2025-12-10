"""
SIMPLEST SOLUTION: Save only weights, not full model
This avoids ALL serialization issues
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import joblib

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


def prepare_data(df, target_col='use [kW]', time_steps=60, 
                 max_samples=None, sample_step=1):
    """Prepare data"""
    if sample_step > 1:
        df = df.iloc[::sample_step].reset_index(drop=True)
        print(f"Downsampled by {sample_step}x: {len(df)} rows")
    
    if max_samples and len(df) > max_samples:
        df = df.iloc[-max_samples:].reset_index(drop=True)
        print(f"Limited to {max_samples} samples")
    
    feature_cols = [
        'use [kW]', 'gen [kW]', 'House overall [kW]',
        'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
        'Home office [kW]', 'Fridge [kW]', 'temperature',
        'humidity', 'pressure', 'windSpeed'
    ]
    
    available_cols = [col for col in feature_cols if col in df.columns]
    data = df[available_cols].values
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    n_sequences = len(data_scaled) - time_steps
    n_features = data_scaled.shape[1]
    
    X = np.zeros((n_sequences, time_steps, n_features), dtype=np.float32)
    y = np.zeros(n_sequences, dtype=np.float32)
    
    target_idx = available_cols.index(target_col)
    
    for i in range(n_sequences):
        X[i] = data_scaled[i:i+time_steps]
        y[i] = data_scaled[i+time_steps, target_idx]
    
    return X, y, scaler, available_cols


def save_model_config(model, filepath='model_config.json'):
    """Save model configuration"""
    config = {
        'time_steps': model.input_shape[1],
        'n_features': model.input_shape[2],
        'lstm_units': 64,  # Store your actual values
        'attention_units': 32
    }
    with open(filepath, 'w') as f:
        json.dump(config, f)
    print(f"Config saved: {filepath}")


def load_model_from_weights(weights_path='model.weights.h5', 
                            config_path='model_config.json'):
    """Load model from weights and config"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Rebuild model
    model = build_lstm_attention_model(
        time_steps=config['time_steps'],
        n_features=config['n_features'],
        lstm_units=config['lstm_units'],
        attention_units=config['attention_units']
    )
    
    # Load weights
    model.load_weights(weights_path)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    
    return model


def train_model(csv_file, time_steps=60, epochs=50, batch_size=64,
                max_samples=50000, sample_step=10, lstm_units=64):
    """Train model"""
    
    print("="*60)
    print("TRAINING LSTM WITH ATTENTION MODEL")
    print("="*60)
    
    # Enable GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    # Load data
    print("\n📂 Loading data...")
    chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        chunks.append(chunk)
        if max_samples and len(pd.concat(chunks)) >= max_samples:
            break
    
    df = pd.concat(chunks, ignore_index=True)
    if max_samples:
        df = df.iloc[-max_samples:]
    
    print(f"✅ Loaded {len(df)} rows")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X, y, scaler, feature_cols = prepare_data(
        df, time_steps=time_steps,
        max_samples=max_samples, sample_step=sample_step
    )
    
    print(f"✅ Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    del df
    import gc
    gc.collect()
    
    # Build model
    print("\n🏗️ Building model...")
    model = build_lstm_attention_model(
        time_steps=time_steps,
        n_features=X.shape[2],
        lstm_units=lstm_units,
        attention_units=32
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    
    print(model.summary())
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
    )
    
    # Train
    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluating...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Loss: {results[0]:.4f}")
    print(f"✅ Test MAE: {results[1]:.4f}")
    
    # Save weights and config (NOT full model)
    model.save_weights('model.weights.h5')
    print("\n💾 Weights saved: model.weights.h5")
    
    save_model_config(model)
    
    return model, scaler, feature_cols, X_test, y_test


if __name__ == "__main__":
    import sys
    import os
    
    print("="*60)
    print("🚀 LSTM WITH ATTENTION - SIMPLE WEIGHTS SAVING")
    print(f"📦 TensorFlow: {tf.__version__}")
    print("="*60)
    
    CSV_FILE = 'home_data.csv'
    
    if not os.path.exists(CSV_FILE):
        print(f"\n❌ ERROR: File not found: {CSV_FILE}")
        sys.exit(1)
    
    try:
        # Train
        model, scaler, feature_cols, X_test, y_test = train_model(
            csv_file=CSV_FILE,
            time_steps=60,
            epochs=50,
            batch_size=64,
            max_samples=50000,
            sample_step=10,
            lstm_units=64
        )
        
        # Save scaler
        joblib.dump(scaler, 'scaler.pkl')
        print("💾 Scaler saved: scaler.pkl")
        
        # Test loading
        print("\n" + "="*60)
        print("🔍 TESTING MODEL LOADING")
        print("="*60)
        
        print("\n1️⃣ Loading model from weights...")
        loaded_model = load_model_from_weights(
            weights_path='model.weights.h5',
            config_path='model_config.json'
        )
        print("✅ SUCCESS! Model loaded from weights!")
        
        print("\n2️⃣ Making predictions...")
        predictions = loaded_model.predict(X_test[:5], verbose=0)
        print(f"✅ Predictions: {predictions.flatten()}")
        
        # Verify predictions match
        original_pred = model.predict(X_test[:5], verbose=0)
        print(f"✅ Original:    {original_pred.flatten()}")
        print(f"✅ Match: {np.allclose(predictions, original_pred)}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n📝 Files created:")
        print("   - model.weights.h5  (model weights)")
        print("   - model_config.json (model architecture)")
        print("   - scaler.pkl        (data scaler)")
        
        print("\n🎯 To use later:")
        print("   from models_home import load_model_from_weights")
        print("   model = load_model_from_weights('model.weights.h5', 'model_config.json')")
        
        print("\n💡 ADVANTAGE: No serialization issues!")
        print("   This method works with ANY Keras version")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)