import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, auc, precision_recall_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

class UCIHeartDiseaseRetrainer:
    """
    Class để retrain mô hình dự đoán bệnh tim với UCI Heart Disease Dataset
    """
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                alpha=0.01
            )
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.model_scores = {}
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def clean_and_encode_udf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pandas UDF để clean data và encode các categorical values thành numeric
        """
        non_feature_cols = ['id', 'dataset']
        cleaned_df = df.drop(columns=[col for col in non_feature_cols if col in df.columns])
        
        print(f"🧹 Cleaned columns: {list(cleaned_df.columns)}")
        
        def encode_column(col_name, series):
            """Encode từng cột categorical thành numeric"""
            if col_name == 'sex':
                mapped = series.map({'Male': 1, 'Female': 0, 'male': 1, 'female': 0, 'M': 1, 'F': 0, 1: 1, 0: 0})
                return mapped.fillna(series)
            elif col_name == 'cp':
                cp_mapping = {
                    'typical angina': 3, 'atypical angina': 2, 'non-anginal pain': 1,
                    'non-anginal': 1, 'asymptomatic': 0, 'asympt': 0,
                    3: 3, 2: 2, 1: 1, 0: 0
                }
                mapped = series.map(cp_mapping)
                return mapped.fillna(series)
            elif col_name == 'fbs':
                fbs_mapping = {'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 'true': 1, 'false': 0, 1: 1, 0: 0}
                mapped = series.map(fbs_mapping)
                return mapped.fillna(series)
            elif col_name == 'restecg':
                restecg_mapping = {
                    'normal': 0, 'st-t abnormality': 1, 'st-t': 1,
                    'lv hypertrophy': 2, 'lv': 2, 0: 0, 1: 1, 2: 2
                }
                mapped = series.map(restecg_mapping)
                return mapped.fillna(series)
            elif col_name == 'exang':
                exang_mapping = {'TRUE': 1, 'FALSE': 0, True: 1, False: 0, 'true': 1, 'false': 0, 1: 1, 0: 0}
                mapped = series.map(exang_mapping)
                return mapped.fillna(series)
            elif col_name == 'slope':
                slope_mapping = {
                    'upsloping': 0, 'up': 0, 'flat': 1, 'downsloping': 2, 'down': 2,
                    0: 0, 1: 1, 2: 2
                }
                mapped = series.map(slope_mapping)
                return mapped.fillna(series)
            elif col_name == 'thal':
                thal_mapping = {
                    'normal': 3, 'fixed defect': 6, 'fixed': 6,
                    'reversable defect': 7, 'reversible defect': 7,
                    'reversable': 7, 'reversible': 7,
                    3: 3, 6: 6, 7: 7, 1: 6, 2: 7
                }
                mapped = series.map(thal_mapping)
                return mapped.fillna(series)
            else:
                return series
        
        encoded_df = cleaned_df.copy()
        for col in encoded_df.columns:
            original_dtype = encoded_df[col].dtype
            
            if original_dtype == 'object':
                encoded_df[col] = encoded_df[col].astype(str).str.strip().str.lower()
            
            encoded_df[col] = encode_column(col, encoded_df[col])
            
            if original_dtype == 'object':
                print(f"  🔤 Encoded {col}: {original_dtype} → numeric")
        
        return encoded_df
    
    def uci_data_validation_udf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pandas UDF để validate dữ liệu sau khi encode
        """
        def validate_column(col_name, series):
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            if col_name == 'age':
                return numeric_series.clip(0, 120)
            elif col_name == 'sex':
                return numeric_series.clip(0, 1)
            elif col_name == 'cp':
                return numeric_series.clip(0, 3)
            elif col_name == 'trestbps':
                return numeric_series.clip(80, 200)
            elif col_name == 'chol':
                return numeric_series.clip(100, 600)
            elif col_name == 'fbs':
                return numeric_series.clip(0, 1)
            elif col_name == 'restecg':
                return numeric_series.clip(0, 2)
            elif col_name == 'thalch':
                return numeric_series.clip(60, 220)
            elif col_name == 'exang':
                return numeric_series.clip(0, 1)
            elif col_name == 'oldpeak':
                return numeric_series.clip(0, 10)
            elif col_name == 'slope':
                return numeric_series.clip(0, 2)
            elif col_name == 'ca':
                return numeric_series.clip(0, 3)
            elif col_name == 'thal':
                numeric_series = numeric_series.replace({1: 6, 2: 7})
                return numeric_series.clip(3, 7)
            elif col_name == 'num':
                return (numeric_series > 0).astype(int)
            else:
                return numeric_series
        
        validated_df = df.copy()
        for col in validated_df.columns:
            validated_df[col] = validate_column(col, validated_df[col])
            
            nan_count = validated_df[col].isna().sum()
            if nan_count > 0:
                print(f"  ⚠️ Warning: {col} has {nan_count} NaN values after validation")
                validated_df[col] = validated_df[col].fillna(validated_df[col].median())
        
        return validated_df
    
    def uci_feature_engineering_udf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pandas UDF để tạo features mới
        """
        def create_heart_features(row):
            features = row.copy()
            
            if 'age' in row.index:
                age = row['age']
                features['age_group'] = 0 if age < 45 else (1 if age < 55 else (2 if age < 65 else 3))
                features['elderly'] = 1 if age >= 65 else 0
            
            if 'chol' in row.index:
                chol = row['chol']
                features['chol_risk'] = 0 if chol < 200 else (1 if chol < 240 else 2)
                features['high_chol'] = 1 if chol >= 240 else 0
            
            if 'trestbps' in row.index:
                bp = row['trestbps']
                features['bp_category'] = 0 if bp < 120 else (1 if bp < 140 else 2)
                features['hypertension'] = 1 if bp >= 140 else 0
            
            if 'thalch' in row.index and 'age' in row.index:
                max_hr = row['thalch']
                age = row['age']
                predicted_max = 220 - age
                features['hr_reserve'] = max_hr / predicted_max if predicted_max > 0 else 0
                features['low_hr_reserve'] = 1 if features['hr_reserve'] < 0.8 else 0
            
            if 'cp' in row.index:
                cp = row['cp']
                features['typical_angina'] = 1 if cp == 3 else 0
                features['asymptomatic'] = 1 if cp == 0 else 0
            
            if 'exang' in row.index and 'oldpeak' in row.index:
                features['exercise_intolerance'] = 1 if (row['exang'] == 1 and row['oldpeak'] > 1) else 0
            
            if 'oldpeak' in row.index:
                oldpeak = row['oldpeak']
                features['significant_st_depression'] = 1 if oldpeak > 2 else 0
            
            if all(col in row.index for col in ['sex', 'age', 'chol', 'trestbps']):
                risk_score = 0
                if row['sex'] == 1:
                    risk_score += 1
                if row['age'] > 55:
                    risk_score += 1
                if row['chol'] > 240:
                    risk_score += 1
                if row['trestbps'] > 140:
                    risk_score += 1
                features['total_risk_factors'] = risk_score
                features['high_risk'] = 1 if risk_score >= 3 else 0
            
            if 'thal' in row.index:
                thal = row['thal']
                features['thal_normal'] = 1 if thal == 3 else 0
                features['thal_defect'] = 1 if thal in [6, 7] else 0
            
            return features
        
        enhanced_df = df.apply(create_heart_features, axis=1, result_type='expand')
        return enhanced_df
    
    def load_kaggle_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset
        """
        print("📂 Loading Heart Disease dataset...")
        df = pd.read_csv(filepath)
        
        print(f"📊 Raw dataset shape: {df.shape}")
        print(f"📋 Raw columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"\n📋 Sample data:")
            print(df.head(3).to_string())
        
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        print("\n🔍 Categorical values preview:")
        for col in categorical_cols:
            if col in df.columns:
                unique_vals = df[col].unique()[:5]
                print(f"  {col}: {unique_vals}")
        
        print(f"\n🎯 Target distribution:")
        if 'num' in df.columns:
            print(df['num'].value_counts())
        
        return df
    
    def retrain_models(self, df: pd.DataFrame, target_column: str = 'num', 
                      test_size: float = 0.2, apply_feature_engineering: bool = True):
        """
        Hàm chính để retrain tất cả các mô hình
        """
        print("🔄 Bắt đầu quá trình retrain mô hình...")
        
        # 1. Clean và encode
        print("🔤 Cleaning and encoding...")
        encoded_df = self.clean_and_encode_udf(df.copy())
        
        # 2. Validate
        print("✅ Validating data...")
        validated_df = self.uci_data_validation_udf(encoded_df)
        
        # 3. Feature engineering
        if apply_feature_engineering:
            print("⚙️ Feature engineering...")
            validated_df = self.uci_feature_engineering_udf(validated_df)
        
        # 4. Final numeric conversion
        print("🔢 Final numeric conversion...")
        for col in validated_df.columns:
            validated_df[col] = pd.to_numeric(validated_df[col], errors='coerce')
            if validated_df[col].isna().sum() > 0:
                validated_df[col] = validated_df[col].fillna(validated_df[col].median())
        
        # 5. Split features và target
        X = validated_df.drop(columns=[target_column])
        y = validated_df[target_column]
        
        self.feature_names = X.columns.tolist()
        print(f"🔢 Total features: {len(self.feature_names)}")
        
        # 6. Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📈 Training set: {self.X_train.shape[0]} samples")
        print(f"📉 Test set: {self.X_test.shape[0]} samples")
        
        # 7. Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # 8. Train models
        print("🤖 Training models...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"   📈 Training {name}...")
            
            try:
                if name in ['SVM', 'Logistic Regression', 'Neural Network']:
                    model.fit(X_train_scaled, self.y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='accuracy')
                    cv_auc = cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
                    cv_auc = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
                
                accuracy = accuracy_score(self.y_test, y_pred)
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                
                self.trained_models[name] = model
                self.model_scores[name] = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'cv_auc_mean': cv_auc.mean(),
                    'cv_auc_std': cv_auc.std()
                }
                
                print(f"   ✅ {name}: Acc={accuracy:.4f}, AUC={auc_score:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
        
        print("\n🎉 Training completed!")
        return self.get_results_summary()
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Summary kết quả
        """
        results = []
        for model_name, scores in self.model_scores.items():
            results.append({
                'Model': model_name,
                'Test Accuracy': scores['accuracy'],
                'Test AUC': scores['auc'],
                'CV Accuracy': f"{scores['cv_accuracy_mean']:.4f}±{scores['cv_accuracy_std']:.4f}",
                'CV AUC': f"{scores['cv_auc_mean']:.4f}±{scores['cv_auc_std']:.4f}",
                'Rank Score': scores['accuracy'] * 0.5 + scores['auc'] * 0.5
            })
        
        results_df = pd.DataFrame(results)
        return results_df.sort_values('Rank Score', ascending=False)


class RandomForestReporter:
    """
    Class để tạo báo cáo chi tiết cho Random Forest model
    """
    
    def __init__(self, retrainer, model_name='Random Forest'):
        self.retrainer = retrainer
        self.model_name = model_name
        self.model = retrainer.trained_models.get(model_name)
        self.scores = retrainer.model_scores.get(model_name)
        
    def generate_comprehensive_report(self):
        """
        Tạo báo cáo toàn diện
        """
        report = "🏥 RANDOM FOREST MODEL - COMPREHENSIVE ANALYSIS REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += self._get_model_overview()
        report += self._get_performance_metrics()
        report += self._get_feature_importance_analysis()
        report += self._get_model_characteristics()
        report += self._get_clinical_insights()
        report += self._get_recommendations()
        
        return report
    
    def _get_model_overview(self):
        overview = "📊 MODEL OVERVIEW\n"
        overview += "-" * 30 + "\n"
        overview += f"Model Type: Random Forest Classifier\n"
        overview += f"Algorithm: Ensemble of Decision Trees\n"
        overview += f"Training Features: {len(self.retrainer.feature_names)}\n"
        
        if self.model:
            overview += f"Model Parameters:\n"
            overview += f"  - n_estimators: {self.model.n_estimators}\n"
            overview += f"  - max_depth: {self.model.max_depth}\n"
            overview += f"  - min_samples_split: {self.model.min_samples_split}\n"
            overview += f"  - min_samples_leaf: {self.model.min_samples_leaf}\n"
        overview += "\n"
        return overview
    
    def _get_performance_metrics(self):
        metrics = "📈 PERFORMANCE METRICS\n"
        metrics += "-" * 30 + "\n"
        
        if self.scores:
            metrics += f"Test Set Performance:\n"
            metrics += f"  🎯 Accuracy: {self.scores['accuracy']:.4f} ({self.scores['accuracy']*100:.2f}%)\n"
            metrics += f"  📊 AUC-ROC: {self.scores['auc']:.4f}\n"
            metrics += f"  🔄 CV Accuracy: {self.scores['cv_accuracy_mean']:.4f} ± {self.scores['cv_accuracy_std']:.4f}\n"
            metrics += f"  🔄 CV AUC: {self.scores['cv_auc_mean']:.4f} ± {self.scores['cv_auc_std']:.4f}\n\n"
            
            # Performance interpretation
            acc = self.scores['accuracy']
            if acc >= 0.90:
                metrics += "  ✅ Excellent performance (>90%)\n"
            elif acc >= 0.85:
                metrics += "  ✅ Very good performance (85-90%)\n"
            elif acc >= 0.80:
                metrics += "  ⚡ Good performance (80-85%)\n"
            else:
                metrics += "  ⚠️ Needs improvement (<80%)\n"
            
            cv_std = self.scores['cv_accuracy_std']
            if cv_std <= 0.02:
                metrics += "  🔒 Very stable model (CV std ≤ 2%)\n"
            elif cv_std <= 0.05:
                metrics += "  🔒 Stable model (CV std ≤ 5%)\n"
            else:
                metrics += "  ⚠️ Model shows variance (CV std > 5%)\n"
        
        metrics += "\n"
        return metrics
    
    def _get_feature_importance_analysis(self):
        importance = "🔍 FEATURE IMPORTANCE ANALYSIS\n"
        importance += "-" * 30 + "\n"
        
        if self.model and hasattr(self.model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.retrainer.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance += "Top 10 Most Important Features:\n"
            for i, (_, row) in enumerate(feature_imp.head(10).iterrows(), 1):
                percentage = row['importance'] * 100
                importance += f"  {i:2d}. {row['feature']:<25}: {percentage:5.2f}%\n"
            
            # Feature categories
            clinical_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                               'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            engineered_features = [f for f in feature_imp['feature'] if f not in clinical_features]
            
            clinical_importance = feature_imp[feature_imp['feature'].isin(clinical_features)]['importance'].sum()
            engineered_importance = feature_imp[feature_imp['feature'].isin(engineered_features)]['importance'].sum()
            
            importance += f"\nFeature Categories:\n"
            importance += f"  📋 Clinical Features: {clinical_importance*100:.1f}% total importance\n"
            importance += f"  ⚙️ Engineered Features: {engineered_importance*100:.1f}% total importance\n"
            
            # Key insights
            top_feature = feature_imp.iloc[0]
            importance += f"\n🏆 Most predictive: {top_feature['feature']} ({top_feature['importance']*100:.1f}%)\n"
        
        importance += "\n"
        return importance
    
    def _get_model_characteristics(self):
        characteristics = "🔧 MODEL CHARACTERISTICS\n"
        characteristics += "-" * 30 + "\n"
        
        characteristics += "Random Forest Advantages:\n"
        characteristics += "  ✅ Handles mixed data types\n"
        characteristics += "  ✅ Robust to outliers\n"
        characteristics += "  ✅ Feature importance rankings\n"
        characteristics += "  ✅ Reduced overfitting\n"
        characteristics += "  ✅ No feature scaling needed\n"
        characteristics += "  ✅ Captures non-linear patterns\n\n"
        
        characteristics += "Considerations:\n"
        characteristics += "  ⚠️ Less interpretable than single tree\n"
        characteristics += "  ⚠️ Can be memory intensive\n"
        characteristics += "  ⚠️ May overfit with noisy data\n\n"
        
        return characteristics
    
    def _get_clinical_insights(self):
        insights = "🏥 CLINICAL INSIGHTS\n"
        insights += "-" * 30 + "\n"
        
        if self.model and hasattr(self.model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.retrainer.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            insights += "Medical Interpretation of Key Predictors:\n\n"
            
            top_5 = feature_imp.head(5)
            for _, row in top_5.iterrows():
                feature = row['feature']
                importance = row['importance'] * 100
                
                insights += f"🔹 {feature} ({importance:.1f}%):\n"
                
                if 'age' in feature.lower():
                    insights += "   Age is a primary cardiovascular risk factor\n"
                elif 'cp' in feature.lower() or 'chest' in feature.lower():
                    insights += "   Chest pain patterns are diagnostic indicators\n"
                elif 'thalch' in feature.lower():
                    insights += "   Maximum heart rate indicates cardiac capacity\n"
                elif 'oldpeak' in feature.lower():
                    insights += "   ST depression suggests coronary ischemia\n"
                elif 'ca' in feature.lower():
                    insights += "   Vessel blockage severity assessment\n"
                elif 'thal' in feature.lower():
                    insights += "   Blood flow and perfusion indicators\n"
                elif 'chol' in feature.lower():
                    insights += "   Cholesterol impacts cardiovascular risk\n"
                elif 'sex' in feature.lower():
                    insights += "   Gender-specific risk patterns\n"
                else:
                    insights += "   Engineered risk factor combination\n"
                insights += "\n"
        
        return insights
    
    def _get_recommendations(self):
        recommendations = "💡 RECOMMENDATIONS\n"
        recommendations += "-" * 30 + "\n"
        
        if self.scores and self.scores['accuracy'] >= 0.85:
            recommendations += "✅ RECOMMENDED FOR CLINICAL USE\n"
            recommendations += "  - High accuracy for screening\n"
            recommendations += "  - Suitable for risk stratification\n"
        elif self.scores and self.scores['accuracy'] >= 0.80:
            recommendations += "⚡ USE WITH CLINICAL OVERSIGHT\n"
            recommendations += "  - Good performance with supervision\n"
            recommendations += "  - Regular monitoring required\n"
        else:
            recommendations += "⚠️ REQUIRES IMPROVEMENT\n"
            recommendations += "  - Below clinical standards\n"
            recommendations += "  - Consider model enhancement\n"
        
        recommendations += "\nImplementation Guidelines:\n"
        recommendations += "  1. 🔄 Regular model retraining\n"
        recommendations += "  2. 👨‍⚕️ Always require physician review\n"
        recommendations += "  3. 📊 Monitor performance metrics\n"
        recommendations += "  4. 🔒 Ensure data privacy compliance\n"
        recommendations += "  5. 📈 Track feature drift\n\n"
        
        return recommendations
    
    def create_visualization_report(self):
        """
        Tạo báo cáo visualization
        """
        if not self.model or self.retrainer.X_test is None:
            print("❌ Model hoặc test data không có!")
            return
        
        # Set up plots
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest Model Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Feature Importance
        feature_imp = pd.DataFrame({
            'feature': self.retrainer.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[0,0].barh(range(len(feature_imp)), feature_imp['importance'], color='skyblue')
        axes[0,0].set_yticks(range(len(feature_imp)))
        axes[0,0].set_yticklabels(feature_imp['feature'])
        axes[0,0].set_xlabel('Feature Importance')
        axes[0,0].set_title('Top 10 Feature Importances')
        axes[0,0].grid(axis='x', alpha=0.3)
        
        # 2. ROC Curve
        y_pred_proba = self.model.predict_proba(self.retrainer.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.retrainer.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
        axes[0,1].grid(alpha=0.3)
        
        # 3. Confusion Matrix
        y_pred = self.model.predict(self.retrainer.X_test)
        cm = confusion_matrix(self.retrainer.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_xlabel('Predicted Label')
        axes[1,0].set_ylabel('True Label')
        axes[1,0].set_title('Confusion Matrix')
        
        # 4. Model Performance Metrics
        metrics_data = {
            'Metric': ['Accuracy', 'AUC', 'CV Accuracy', 'CV AUC'],
            'Value': [
                self.scores['accuracy'],
                self.scores['auc'], 
                self.scores['cv_accuracy_mean'],
                self.scores['cv_auc_mean']
            ]
        }
        
        bars = axes[1,1].bar(metrics_data['Metric'], metrics_data['Value'], 
                           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Model Performance Metrics')
        axes[1,1].set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_data['Value']):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed classification report
        print("\n📊 DETAILED CLASSIFICATION REPORT:")
        print("=" * 50)
        print(classification_report(self.retrainer.y_test, y_pred, target_names=['No Disease', 'Disease']))

    def create_additional_plots(self):
        """
        Tạo thêm các plots chi tiết
        """
        if not self.model or self.retrainer.X_test is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Additional Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Precision-Recall Curve
        y_pred_proba = self.model.predict_proba(self.retrainer.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self.retrainer.y_test, y_pred_proba)
        
        axes[0,0].plot(recall, precision, color='blue', lw=2)
        axes[0,0].set_xlabel('Recall')
        axes[0,0].set_ylabel('Precision')
        axes[0,0].set_title('Precision-Recall Curve')
        axes[0,0].grid(alpha=0.3)
        
        # 2. Feature Importance Pie Chart
        feature_imp = pd.DataFrame({
            'feature': self.retrainer.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_imp.head(8)
        others_importance = feature_imp.iloc[8:]['importance'].sum()
        
        plot_data = list(top_features['importance']) + [others_importance]
        plot_labels = list(top_features['feature']) + ['Others']
        
        axes[0,1].pie(plot_data, labels=plot_labels, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Feature Importance Distribution')
        
        # 3. Prediction Probability Distribution
        y_pred_proba = self.model.predict_proba(self.retrainer.X_test)[:, 1]
        
        # Separate by actual class
        prob_no_disease = y_pred_proba[self.retrainer.y_test == 0]
        prob_disease = y_pred_proba[self.retrainer.y_test == 1]
        
        axes[1,0].hist(prob_no_disease, bins=20, alpha=0.7, label='No Disease', color='blue')
        axes[1,0].hist(prob_disease, bins=20, alpha=0.7, label='Disease', color='red')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Prediction Probability Distribution')
        axes[1,0].legend()
        axes[1,0].grid(alpha=0.3)
        
        # 4. Model Comparison
        model_comparison = []
        for name, scores in self.retrainer.model_scores.items():
            model_comparison.append({
                'Model': name,
                'Accuracy': scores['accuracy'],
                'AUC': scores['auc']
            })
        
        comparison_df = pd.DataFrame(model_comparison)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=True)
        
        x_pos = np.arange(len(comparison_df))
        axes[1,1].barh(x_pos, comparison_df['Accuracy'], alpha=0.7, label='Accuracy')
        axes[1,1].barh(x_pos, comparison_df['AUC'], alpha=0.7, label='AUC')
        axes[1,1].set_yticks(x_pos)
        axes[1,1].set_yticklabels(comparison_df['Model'])
        axes[1,1].set_xlabel('Score')
        axes[1,1].set_title('Model Performance Comparison')
        axes[1,1].legend()
        axes[1,1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def export_model_report_to_file(self, filename='random_forest_report.txt'):
        """
        Xuất báo cáo ra file
        """
        report = self.generate_comprehensive_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
            
            # Add additional statistics
            f.write("\n" + "="*70 + "\n")
            f.write("ADDITIONAL STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            if self.retrainer.X_test is not None:
                y_pred = self.model.predict(self.retrainer.X_test)
                y_pred_proba = self.model.predict_proba(self.retrainer.X_test)[:, 1]
                
                # Classification report
                f.write("Classification Report:\n")
                f.write("-" * 30 + "\n")
                f.write(classification_report(self.retrainer.y_test, y_pred, 
                                           target_names=['No Disease', 'Disease']))
                f.write("\n")
                
                # Confusion Matrix Details
                cm = confusion_matrix(self.retrainer.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                f.write("Confusion Matrix Details:\n")
                f.write("-" * 30 + "\n")
                f.write(f"True Negatives (TN): {tn}\n")
                f.write(f"False Positives (FP): {fp}\n")
                f.write(f"False Negatives (FN): {fn}\n")
                f.write(f"True Positives (TP): {tp}\n\n")
                
                # Calculate additional metrics
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                f.write("Additional Metrics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")
                f.write(f"Specificity: {specificity:.4f}\n")
                f.write(f"Positive Predictive Value: {ppv:.4f}\n")
                f.write(f"Negative Predictive Value: {npv:.4f}\n\n")
                
                # Feature importance table
                feature_imp = pd.DataFrame({
                    'Feature': self.retrainer.feature_names,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                f.write("Complete Feature Importance Table:\n")
                f.write("-" * 50 + "\n")
                for i, (_, row) in enumerate(feature_imp.iterrows(), 1):
                    f.write(f"{i:2d}. {row['Feature']:<30}: {row['Importance']:.6f}\n")
        
        print(f"📄 Report exported to {filename}")


def generate_complete_analysis(filepath: str):
    """
    Hàm chính để chạy toàn bộ analysis
    """
    print("🚀 Starting Complete Heart Disease Analysis...")
    print("=" * 60)
    
    # 1. Initialize and train models
    retrainer = UCIHeartDiseaseRetrainer()
    df = retrainer.load_kaggle_dataset(filepath)
    
    print("\n🔬 Training all models...")
    results = retrainer.retrain_models(df, apply_feature_engineering=True)
    
    # 2. Display all model results
    print("\n📊 ALL MODELS PERFORMANCE:")
    print("=" * 60)
    print(results.to_string(index=False))
    
    # 3. Generate Random Forest detailed report
    print("\n🌲 GENERATING RANDOM FOREST DETAILED REPORT...")
    print("=" * 60)
    
    rf_reporter = RandomForestReporter(retrainer)
    
    # Text report
    text_report = rf_reporter.generate_comprehensive_report()
    print(text_report)
    
    # 4. Create visualizations
    print("📈 Creating visualizations...")
    rf_reporter.create_visualization_report()
    
    print("\n📊 Creating additional plots...")
    rf_reporter.create_additional_plots()
    
    # 5. Export detailed report to file
    print("\n💾 Exporting detailed report...")
    rf_reporter.export_model_report_to_file('heart_disease_rf_report.txt')
    
    # 6. Save trained models
    print("\n💾 Saving trained models...")
    for name, model in retrainer.trained_models.items():
        filename = f"model_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"   Saved {name} to {filename}")
    
    # Save scaler
    joblib.dump(retrainer.scaler, 'scaler.pkl')
    print("   Saved scaler to scaler.pkl")
    
    # 7. Generate prediction examples
    print("\n🔮 PREDICTION EXAMPLES:")
    print("=" * 40)
    
    # Create sample prediction data
    sample_data = df.iloc[:5].drop(columns=['num'] + 
                                  (['id', 'dataset'] if 'id' in df.columns else []))
    
    # Predict with Random Forest
    rf_model = retrainer.trained_models['Random Forest']
    
    # Process sample data same way as training
    processed_sample = retrainer.clean_and_encode_udf(sample_data.copy())
    processed_sample = retrainer.uci_data_validation_udf(processed_sample)
    processed_sample = retrainer.uci_feature_engineering_udf(processed_sample)
    
    # Ensure same features
    processed_sample = processed_sample[retrainer.feature_names]
    
    # Make predictions
    predictions = rf_model.predict(processed_sample)
    probabilities = rf_model.predict_proba(processed_sample)[:, 1]
    
    print("Sample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        risk_level = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.3 else "LOW")
        print(f"  Patient {i+1}: {'Disease' if pred == 1 else 'No Disease'} "
              f"(Probability: {prob:.3f}, Risk: {risk_level})")
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("  📄 heart_disease_rf_report.txt - Detailed text report")
    print("  💾 model_*.pkl - Trained model files")
    print("  💾 scaler.pkl - Feature scaler")
    print("\nRecommendation: Use Random Forest model for best performance!")
    
    return retrainer, rf_reporter


def load_and_analyze_custom_data(filepath: str):
    """
    Wrapper function để analyze dữ liệu custom
    """
    try:
        return generate_complete_analysis(filepath)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your data file format and path.")
        return None, None


# Demo function with sample data
def demo_with_sample_data():
    """
    Demo với dữ liệu mẫu nếu không có file
    """
    print("🔬 Creating sample data for demonstration...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'cp': np.random.choice(['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'], n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(150, 350, n_samples),
        'fbs': np.random.choice(['TRUE', 'FALSE'], n_samples),
        'restecg': np.random.choice(['normal', 'st-t abnormality', 'lv hypertrophy'], n_samples),
        'thalch': np.random.randint(80, 200, n_samples),
        'exang': np.random.choice(['TRUE', 'FALSE'], n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.choice(['upsloping', 'flat', 'downsloping'], n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.choice(['normal', 'fixed defect', 'reversable defect'], n_samples),
        'num': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_heart_data.csv', index=False)
    
    print("📁 Sample data saved to 'sample_heart_data.csv'")
    return generate_complete_analysis('sample_heart_data.csv')


if __name__ == "__main__":
    import sys
    
    print("🏥 HEART DISEASE PREDICTION - COMPLETE ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Check if file path provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"📂 Loading data from: {filepath}")
        retrainer, reporter = load_and_analyze_custom_data(filepath)
    else:
        print("📄 No file path provided. Options:")
        print("  1. Run with file: python script.py your_data.csv")
        print("  2. Run demo with sample data")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            filepath = input("Enter path to your CSV file: ").strip()
            retrainer, reporter = load_and_analyze_custom_data(filepath)
        else:
            print("\n🎯 Running demo with sample data...")
            retrainer, reporter = demo_with_sample_data()
    
    if retrainer and reporter:
        print("\n✅ Analysis completed successfully!")
        print("📊 Check the generated visualizations and report file.")
    else:
        print("\n❌ Analysis failed. Please check your data and try again.")