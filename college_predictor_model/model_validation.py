import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelValidator:
    """
    Comprehensive validation system for the college predictor model
    with year-wise validation and accuracy assessment
    """
    
    def __init__(self, data_path: str = 'features_iit_ranks.csv'):
        """
        Initialize validator with feature-engineered data
        """
        self.data = pd.read_csv(data_path)
        self.validation_results = {}
        
    def prepare_validation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for year-wise validation
        Split data by years for temporal validation
        
        Returns:
            train_data, validation_data, test_data
        """
        # Sort by year
        sorted_data = self.data.sort_values('year')
        
        # Split by years (2022 for training, 2023 for validation, 2024 for testing)
        train_data = sorted_data[sorted_data['year'] == 2022].copy()
        val_data = sorted_data[sorted_data['year'] == 2023].copy()
        test_data = sorted_data[sorted_data['year'] == 2024].copy()
        
        print(f"Training data (2022): {len(train_data)} records")
        print(f"Validation data (2023): {len(val_data)} records")
        print(f"Test data (2024): {len(test_data)} records")
        
        return train_data, val_data, test_data
    
    def create_eligibility_labels(self, data: pd.DataFrame, test_ranks: List[int]) -> pd.DataFrame:
        """
        Create eligibility labels for different test ranks
        
        Args:
            data: DataFrame with rank data
            test_ranks: List of ranks to test eligibility for
        
        Returns:
            DataFrame with eligibility labels
        """
        labeled_data = []
        
        for _, row in data.iterrows():
            for rank in test_ranks:
                # Create a record for each rank-program combination
                record = row.copy()
                record['test_rank'] = rank
                record['eligible'] = 1 if rank <= row['closing_rank'] else 0
                labeled_data.append(record)
        
        return pd.DataFrame(labeled_data)
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training
        
        Returns:
            X (features), y (labels)
        """
        # Select relevant features
        feature_columns = ['test_rank', 'opening_rank', 'closing_rank', 'rank_range', 'program_popularity']
        
        # Add encoded categorical features
        data_encoded = data.copy()
        
        # Encode seat_type
        seat_type_dummies = pd.get_dummies(data['seat_type'], prefix='seat')
        data_encoded = pd.concat([data_encoded, seat_type_dummies], axis=1)
        
        # Encode gender
        gender_dummies = pd.get_dummies(data['gender'], prefix='gender')
        data_encoded = pd.concat([data_encoded, gender_dummies], axis=1)
        
        # Encode institute (top institutes only to avoid overfitting)
        top_institutes = data['institute_name'].value_counts().head(10).index
        for institute in top_institutes:
            data_encoded[f'institute_{institute.replace(" ", "_")}'] = (data['institute_name'] == institute).astype(int)
        
        # Select all feature columns
        feature_cols = feature_columns + [col for col in data_encoded.columns if col.startswith(('seat_', 'gender_', 'institute_'))]
        feature_cols = [col for col in feature_cols if col in data_encoded.columns]
        
        X = data_encoded[feature_cols].fillna(0)
        y = data_encoded['eligible']
        
        return X.values, y.values
    
    def train_and_validate_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                                test_data: pd.DataFrame) -> Dict:
        """
        Train model on training data and validate on validation/test sets
        
        Returns:
            Dictionary with validation results
        """
        # Define test ranks for creating labels
        test_ranks = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 50000]
        
        # Create labeled datasets
        print("Creating labeled datasets...")
        train_labeled = self.create_eligibility_labels(train_data, test_ranks)
        val_labeled = self.create_eligibility_labels(val_data, test_ranks)
        test_labeled = self.create_eligibility_labels(test_data, test_ranks)
        
        # Prepare features
        print("Preparing features...")
        X_train, y_train = self.prepare_features(train_labeled)
        X_val, y_val = self.prepare_features(val_labeled)
        X_test, y_test = self.prepare_features(test_labeled)
        
        # Train model
        print("Training model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Validate on different sets
        results = {}
        
        # Training accuracy
        train_pred = model.predict(X_train)
        results['train_accuracy'] = accuracy_score(y_train, train_pred)
        
        # Validation accuracy
        val_pred = model.predict(X_val)
        results['val_accuracy'] = accuracy_score(y_val, val_pred)
        
        # Test accuracy
        test_pred = model.predict(X_test)
        results['test_accuracy'] = accuracy_score(y_test, test_pred)
        
        # Detailed metrics for test set
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')
        results['test_precision'] = precision
        results['test_recall'] = recall
        results['test_f1'] = f1
        
        # Classification report
        results['classification_report'] = classification_report(y_test, test_pred)
        
        # Feature importance
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        results['model'] = model
        
        return results
    
    def validate_rank_predictions(self, model, test_data: pd.DataFrame) -> Dict:
        """
        Validate model predictions for specific rank ranges
        
        Returns:
            Dictionary with rank-wise validation results
        """
        rank_ranges = [
            (1, 1000, "Top 1K"),
            (1001, 5000, "1K-5K"),
            (5001, 10000, "5K-10K"),
            (10001, 20000, "10K-20K"),
            (20001, 50000, "20K-50K")
        ]
        
        rank_results = {}
        
        for min_rank, max_rank, label in rank_ranges:
            # Test ranks in this range
            test_ranks = list(range(min_rank, min(max_rank + 1, 50001), 1000))
            
            # Create labeled data for this range
            labeled_data = self.create_eligibility_labels(test_data, test_ranks)
            
            if len(labeled_data) > 0:
                X, y = self.prepare_features(labeled_data)
                predictions = model.predict(X)
                
                accuracy = accuracy_score(y, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted')
                
                rank_results[label] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'sample_size': len(labeled_data)
                }
        
        return rank_results
    
    def cross_validate_model(self, data: pd.DataFrame) -> Dict:
        """
        Perform cross-validation on the entire dataset
        
        Returns:
            Cross-validation results
        """
        # Create labeled data
        test_ranks = [1000, 5000, 10000, 20000, 30000]
        labeled_data = self.create_eligibility_labels(data, test_ranks)
        
        # Prepare features
        X, y = self.prepare_features(labeled_data)
        
        # Perform cross-validation
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for faster CV
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def run_comprehensive_validation(self) -> Dict:
        """
        Run comprehensive validation including year-wise and cross-validation
        
        Returns:
            Complete validation results
        """
        print("Starting comprehensive model validation...")
        print("=" * 50)
        
        # Prepare data
        train_data, val_data, test_data = self.prepare_validation_data()
        
        # Year-wise validation
        print("\n1. Year-wise validation...")
        year_wise_results = self.train_and_validate_model(train_data, val_data, test_data)
        
        # Rank-wise validation
        print("\n2. Rank-wise validation...")
        rank_wise_results = self.validate_rank_predictions(year_wise_results['model'], test_data)
        
        # Cross-validation
        print("\n3. Cross-validation...")
        cv_results = self.cross_validate_model(self.data)
        
        # Compile all results
        comprehensive_results = {
            'year_wise_validation': {
                'train_accuracy': year_wise_results['train_accuracy'],
                'val_accuracy': year_wise_results['val_accuracy'],
                'test_accuracy': year_wise_results['test_accuracy'],
                'test_precision': year_wise_results['test_precision'],
                'test_recall': year_wise_results['test_recall'],
                'test_f1': year_wise_results['test_f1']
            },
            'rank_wise_validation': rank_wise_results,
            'cross_validation': cv_results,
            'feature_importance': year_wise_results['feature_importance'].head(10).to_dict('records'),
            'model_ready_for_production': self._assess_production_readiness(year_wise_results, cv_results)
        }
        
        # Save model if it passes validation
        if comprehensive_results['model_ready_for_production']:
            joblib.dump(year_wise_results['model'], 'validated_college_predictor.joblib')
            print("\n✅ Model saved as 'validated_college_predictor.joblib'")
        
        return comprehensive_results
    
    def _assess_production_readiness(self, year_wise_results: Dict, cv_results: Dict) -> bool:
        """
        Assess if model is ready for production based on validation metrics
        
        Returns:
            Boolean indicating production readiness
        """
        # Define minimum thresholds
        min_accuracy = 0.85
        min_precision = 0.80
        min_recall = 0.80
        max_overfitting_gap = 0.10
        
        # Check accuracy
        test_accuracy = year_wise_results['test_accuracy']
        if test_accuracy < min_accuracy:
            return False
        
        # Check precision and recall
        if (year_wise_results['test_precision'] < min_precision or 
            year_wise_results['test_recall'] < min_recall):
            return False
        
        # Check for overfitting
        train_test_gap = year_wise_results['train_accuracy'] - year_wise_results['test_accuracy']
        if train_test_gap > max_overfitting_gap:
            return False
        
        # Check cross-validation consistency
        if cv_results['cv_std'] > 0.05:  # High variance in CV scores
            return False
        
        return True
    
    def print_validation_summary(self, results: Dict):
        """
        Print a comprehensive validation summary
        """
        print("\n" + "=" * 60)
        print("MODEL VALIDATION SUMMARY")
        print("=" * 60)
        
        # Year-wise results
        yw = results['year_wise_validation']
        print(f"\n📊 Year-wise Validation:")
        print(f"   Training Accuracy:   {yw['train_accuracy']:.3f}")
        print(f"   Validation Accuracy: {yw['val_accuracy']:.3f}")
        print(f"   Test Accuracy:       {yw['test_accuracy']:.3f}")
        print(f"   Test Precision:      {yw['test_precision']:.3f}")
        print(f"   Test Recall:         {yw['test_recall']:.3f}")
        print(f"   Test F1-Score:       {yw['test_f1']:.3f}")
        
        # Cross-validation results
        cv = results['cross_validation']
        print(f"\n🔄 Cross-validation:")
        print(f"   Mean Accuracy: {cv['cv_mean']:.3f} (±{cv['cv_std']:.3f})")
        
        # Rank-wise results
        print(f"\n📈 Rank-wise Performance:")
        for rank_range, metrics in results['rank_wise_validation'].items():
            print(f"   {rank_range}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
        # Production readiness
        ready = results['model_ready_for_production']
        status = "✅ READY" if ready else "❌ NOT READY"
        print(f"\n🚀 Production Status: {status}")
        
        if ready:
            print("   Model meets all quality thresholds for production deployment.")
        else:
            print("   Model requires improvement before production deployment.")

# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = ModelValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Print summary
    validator.print_validation_summary(results)