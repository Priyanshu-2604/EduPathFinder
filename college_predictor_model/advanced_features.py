import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for IIT college prediction model
    including categorical encoding, historical trends, and program competitiveness
    """
    
    def __init__(self, data_path: str = 'cleaned_iit_ranks.csv'):
        """
        Initialize with cleaned data
        """
        self.data = pd.read_csv(data_path)
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def create_historical_features(self) -> pd.DataFrame:
        """
        Create features based on historical trends and patterns
        
        Returns:
            DataFrame with historical features
        """
        df = self.data.copy()
        
        # Sort by institute, program, seat_type, gender, and year
        df = df.sort_values(['institute_name', 'academic_program_name', 'seat_type', 'gender', 'year'])
        
        # Calculate year-over-year changes in cutoffs
        df['opening_rank_change'] = df.groupby(['institute_name', 'academic_program_name', 'seat_type', 'gender'])['opening_rank'].diff()
        df['closing_rank_change'] = df.groupby(['institute_name', 'academic_program_name', 'seat_type', 'gender'])['closing_rank'].diff()
        
        # Calculate trend indicators
        df['rank_trend'] = np.where(df['closing_rank_change'] > 0, 'increasing', 
                                   np.where(df['closing_rank_change'] < 0, 'decreasing', 'stable'))
        
        # Calculate volatility (standard deviation of ranks over years)
        rank_volatility = df.groupby(['institute_name', 'academic_program_name', 'seat_type', 'gender'])['closing_rank'].std().reset_index()
        rank_volatility.columns = ['institute_name', 'academic_program_name', 'seat_type', 'gender', 'rank_volatility']
        
        df = df.merge(rank_volatility, on=['institute_name', 'academic_program_name', 'seat_type', 'gender'], how='left')
        df['rank_volatility'] = df['rank_volatility'].fillna(0)
        
        # Calculate average historical performance
        historical_avg = df.groupby(['institute_name', 'academic_program_name', 'seat_type', 'gender']).agg({
            'opening_rank': 'mean',
            'closing_rank': 'mean'
        }).reset_index()
        historical_avg.columns = ['institute_name', 'academic_program_name', 'seat_type', 'gender', 
                                'historical_avg_opening', 'historical_avg_closing']
        
        df = df.merge(historical_avg, on=['institute_name', 'academic_program_name', 'seat_type', 'gender'], how='left')
        
        return df
    
    def create_competitiveness_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to program and institute competitiveness
        
        Args:
            df: DataFrame with historical features
        
        Returns:
            DataFrame with competitiveness features
        """
        # Institute-level competitiveness
        institute_stats = df.groupby('institute_name').agg({
            'closing_rank': ['mean', 'min', 'std'],
            'academic_program_name': 'nunique'
        }).reset_index()
        
        institute_stats.columns = ['institute_name', 'institute_avg_closing', 'institute_min_closing', 
                                 'institute_closing_std', 'institute_program_count']
        
        # Institute tier based on average closing rank
        institute_stats['institute_tier'] = pd.cut(institute_stats['institute_avg_closing'], 
                                                  bins=[0, 5000, 15000, float('inf')], 
                                                  labels=['Tier1', 'Tier2', 'Tier3'])
        
        df = df.merge(institute_stats, on='institute_name', how='left')
        
        # Program-level competitiveness across all institutes
        program_stats = df.groupby('academic_program_name').agg({
            'closing_rank': ['mean', 'min', 'std'],
            'institute_name': 'nunique'
        }).reset_index()
        
        program_stats.columns = ['academic_program_name', 'program_avg_closing', 'program_min_closing',
                               'program_closing_std', 'program_institute_count']
        
        df = df.merge(program_stats, on='academic_program_name', how='left')
        
        # Seat type competitiveness
        seat_stats = df.groupby(['seat_type', 'gender']).agg({
            'closing_rank': ['mean', 'std']
        }).reset_index()
        
        seat_stats.columns = ['seat_type', 'gender', 'seat_gender_avg_closing', 'seat_gender_closing_std']
        df = df.merge(seat_stats, on=['seat_type', 'gender'], how='left')
        
        # Relative competitiveness scores
        df['institute_competitiveness'] = df['institute_avg_closing'].rank(ascending=True) / len(df['institute_name'].unique())
        df['program_competitiveness'] = df['program_avg_closing'].rank(ascending=True) / len(df['academic_program_name'].unique())
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create and encode categorical features
        
        Args:
            df: DataFrame with competitiveness features
        
        Returns:
            DataFrame with encoded categorical features
        """
        # Branch categorization based on program names
        def categorize_branch(program_name):
            program_lower = program_name.lower()
            if any(term in program_lower for term in ['computer', 'cs', 'information', 'it']):
                return 'Computer_Science'
            elif any(term in program_lower for term in ['electrical', 'electronics', 'ee', 'ece']):
                return 'Electrical_Electronics'
            elif any(term in program_lower for term in ['mechanical', 'me']):
                return 'Mechanical'
            elif any(term in program_lower for term in ['civil', 'ce']):
                return 'Civil'
            elif any(term in program_lower for term in ['chemical', 'che']):
                return 'Chemical'
            elif any(term in program_lower for term in ['aerospace', 'aeronautical']):
                return 'Aerospace'
            elif any(term in program_lower for term in ['biotechnology', 'biomedical', 'bio']):
                return 'Biotechnology'
            elif any(term in program_lower for term in ['metallurgy', 'materials', 'met']):
                return 'Materials_Metallurgy'
            else:
                return 'Other_Engineering'
        
        df['branch_category'] = df['academic_program_name'].apply(categorize_branch)
        
        # Institute location/region (simplified categorization)
        def categorize_institute_region(institute_name):
            institute_lower = institute_name.lower()
            if any(city in institute_lower for city in ['delhi', 'bombay', 'mumbai', 'kanpur', 'kharagpur', 'chennai', 'madras']):
                return 'Metro_Old_IIT'
            elif any(city in institute_lower for city in ['roorkee', 'guwahati', 'hyderabad', 'indore']):
                return 'Tier1_City_IIT'
            else:
                return 'Other_IIT'
        
        df['institute_region'] = df['institute_name'].apply(categorize_institute_region)
        
        # Create interaction features
        df['branch_seat_interaction'] = df['branch_category'] + '_' + df['seat_type']
        df['region_tier_interaction'] = df['institute_region'] + '_' + df['institute_tier'].astype(str)
        
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using various encoding techniques
        
        Args:
            df: DataFrame with categorical features
        
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        # One-hot encoding for low cardinality categorical variables
        low_cardinality_cols = ['seat_type', 'gender', 'rank_trend', 'institute_tier', 
                               'branch_category', 'institute_region']
        
        for col in low_cardinality_cols:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Label encoding for high cardinality categorical variables
        high_cardinality_cols = ['institute_name', 'academic_program_name', 
                               'branch_seat_interaction', 'region_tier_interaction']
        
        for col in high_cardinality_cols:
            if col in df_encoded.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[f'{col}_encoded'] = self.encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def create_rank_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on rank distributions and percentiles
        
        Args:
            df: DataFrame with encoded features
        
        Returns:
            DataFrame with rank-based features
        """
        # Rank percentiles within each category
        df['opening_rank_percentile'] = df.groupby(['seat_type', 'gender'])['opening_rank'].rank(pct=True)
        df['closing_rank_percentile'] = df.groupby(['seat_type', 'gender'])['closing_rank'].rank(pct=True)
        
        # Rank gap (difference between opening and closing ranks)
        df['rank_gap'] = df['closing_rank'] - df['opening_rank']
        df['rank_gap_ratio'] = df['rank_gap'] / df['closing_rank']
        
        # Normalized ranks (0-1 scale)
        df['opening_rank_normalized'] = df['opening_rank'] / df['opening_rank'].max()
        df['closing_rank_normalized'] = df['closing_rank'] / df['closing_rank'].max()
        
        # Rank position relative to institute average
        df['rank_vs_institute_avg'] = df['closing_rank'] / df['institute_avg_closing']
        df['rank_vs_program_avg'] = df['closing_rank'] / df['program_avg_closing']
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features based on year and trends
        
        Args:
            df: DataFrame with rank-based features
        
        Returns:
            DataFrame with temporal features
        """
        # Years since first year in dataset
        min_year = df['year'].min()
        df['years_since_start'] = df['year'] - min_year
        
        # Cyclical encoding for year (if needed for future predictions)
        df['year_sin'] = np.sin(2 * np.pi * df['year'] / 4)  # Assuming 4-year cycle
        df['year_cos'] = np.cos(2 * np.pi * df['year'] / 4)
        
        # Moving averages (if sufficient historical data)
        df = df.sort_values(['institute_name', 'academic_program_name', 'seat_type', 'gender', 'year'])
        
        # 2-year moving average of closing ranks
        df['closing_rank_ma2'] = df.groupby(['institute_name', 'academic_program_name', 'seat_type', 'gender'])['closing_rank'].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str = None, k: int = 50) -> pd.DataFrame:
        """
        Select best features using statistical tests
        
        Args:
            df: DataFrame with all features
            target_col: Target column for feature selection
            k: Number of top features to select
        
        Returns:
            DataFrame with selected features
        """
        if target_col is None or target_col not in df.columns:
            # If no target specified, return all numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            return df[numeric_cols]
        
        # Prepare features and target
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Select best features
        selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Create result dataframe
        result_df = df[selected_features + [target_col] if target_col else selected_features].copy()
        
        # Store feature scores for analysis
        feature_scores = pd.DataFrame({
            'feature': feature_cols,
            'score': selector.scores_,
            'selected': selector.get_support()
        }).sort_values('score', ascending=False)
        
        print(f"\nTop 10 selected features:")
        print(feature_scores.head(10)[['feature', 'score']].to_string(index=False))
        
        return result_df
    
    def run_advanced_feature_engineering(self, save_path: str = 'advanced_features_iit_ranks.csv') -> pd.DataFrame:
        """
        Run complete advanced feature engineering pipeline
        
        Args:
            save_path: Path to save the final feature-engineered dataset
        
        Returns:
            DataFrame with all advanced features
        """
        print("Starting advanced feature engineering...")
        print("=" * 50)
        
        # Step 1: Create historical features
        print("1. Creating historical trend features...")
        df = self.create_historical_features()
        print(f"   Added historical features. Shape: {df.shape}")
        
        # Step 2: Create competitiveness features
        print("2. Creating competitiveness features...")
        df = self.create_competitiveness_features(df)
        print(f"   Added competitiveness features. Shape: {df.shape}")
        
        # Step 3: Create categorical features
        print("3. Creating categorical features...")
        df = self.create_categorical_features(df)
        print(f"   Added categorical features. Shape: {df.shape}")
        
        # Step 4: Encode categorical variables
        print("4. Encoding categorical variables...")
        df = self.encode_categorical_variables(df)
        print(f"   Encoded categorical variables. Shape: {df.shape}")
        
        # Step 5: Create rank-based features
        print("5. Creating rank-based features...")
        df = self.create_rank_based_features(df)
        print(f"   Added rank-based features. Shape: {df.shape}")
        
        # Step 6: Create temporal features
        print("6. Creating temporal features...")
        df = self.create_temporal_features(df)
        print(f"   Added temporal features. Shape: {df.shape}")
        
        # Step 7: Handle missing values
        print("7. Handling missing values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        print(f"   Handled missing values. Final shape: {df.shape}")
        
        # Save the advanced features dataset
        df.to_csv(save_path, index=False)
        print(f"\n✅ Advanced features saved to: {save_path}")
        
        # Print feature summary
        self._print_feature_summary(df)
        
        return df
    
    def _print_feature_summary(self, df: pd.DataFrame):
        """
        Print summary of created features
        """
        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 50)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Total Features: {len(df.columns)}")
        print(f"   Numeric Features: {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"   Categorical Features: {len(df.select_dtypes(include=['object']).columns)}")
        
        print(f"\n🏗️ Feature Categories:")
        feature_categories = {
            'Historical': [col for col in df.columns if any(term in col for term in ['change', 'trend', 'volatility', 'historical'])],
            'Competitiveness': [col for col in df.columns if any(term in col for term in ['competitiveness', 'tier', 'avg', 'min', 'std'])],
            'Categorical': [col for col in df.columns if any(term in col for term in ['seat_type', 'gender', 'branch', 'region', 'encoded'])],
            'Rank-based': [col for col in df.columns if any(term in col for term in ['rank', 'percentile', 'gap', 'normalized'])],
            'Temporal': [col for col in df.columns if any(term in col for term in ['year', 'since', 'sin', 'cos', 'ma'])]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"   {category}: {len(features)} features")
        
        print(f"\n🎯 Key Features Created:")
        key_features = [
            'rank_volatility', 'institute_competitiveness', 'program_competitiveness',
            'branch_category', 'institute_region', 'rank_gap_ratio', 'closing_rank_percentile'
        ]
        
        for feature in key_features:
            if feature in df.columns:
                print(f"   ✓ {feature}")

# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Run advanced feature engineering
    advanced_df = feature_engineer.run_advanced_feature_engineering()
    
    print(f"\n🚀 Advanced feature engineering completed!")
    print(f"   Dataset ready for advanced model training.")