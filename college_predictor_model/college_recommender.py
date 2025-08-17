import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class CollegeRecommender:
    """
    Main recommendation system for IIT college and branch allocation
    based on student rank, category, gender, and other factors.
    """
    
    def __init__(self, data_path: str = 'features_iit_ranks.csv', model_path: str = 'college_predictor_rf.joblib'):
        """
        Initialize the recommender with data and trained model
        """
        self.data = pd.read_csv(data_path)
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print("Warning: Model file not found. Please train the model first.")
            self.model = None
        
        # Preprocess data for faster lookups
        self._prepare_lookup_tables()
    
    def _prepare_lookup_tables(self):
        """
        Create lookup tables for faster recommendation
        """
        # Create category mappings
        self.seat_type_mapping = {
            'OPEN': ['OPEN', 'GEN'],
            'OBC-NCL': ['OBC-NCL', 'OBC'],
            'SC': ['SC'],
            'ST': ['ST'],
            'EWS': ['EWS']
        }
        
        # Create institute ranking based on historical cutoffs
        institute_stats = self.data.groupby('institute_name').agg({
            'opening_rank': 'mean',
            'closing_rank': 'mean'
        }).reset_index()
        institute_stats['avg_rank'] = (institute_stats['opening_rank'] + institute_stats['closing_rank']) / 2
        self.institute_ranking = institute_stats.sort_values('avg_rank')['institute_name'].tolist()
    
    def get_eligible_programs(self, student_rank: int, category: str, gender: str, 
                            safety_margin: float = 0.1) -> List[Dict]:
        """
        Get all eligible programs for a student based on their profile
        
        Args:
            student_rank: Student's rank
            category: Student's category (OPEN, OBC-NCL, SC, ST, EWS)
            gender: Student's gender (Male, Female)
            safety_margin: Safety margin for rank cutoffs (default 10%)
        
        Returns:
            List of eligible programs with details
        """
        eligible_programs = []
        
        # Get applicable seat types for the category
        applicable_seat_types = self._get_applicable_seat_types(category)
        
        # Filter data based on criteria
        for seat_type in applicable_seat_types:
            filtered_data = self.data[
                (self.data['seat_type'] == seat_type) &
                (self.data['gender'] == gender)
            ].copy()
            
            # Apply safety margin to closing ranks
            safety_rank = student_rank * (1 + safety_margin)
            
            # Find programs where student's rank is within closing rank
            eligible = filtered_data[filtered_data['closing_rank'] >= safety_rank]
            
            for _, row in eligible.iterrows():
                program_info = {
                    'institute_name': row['institute_name'],
                    'academic_program_name': row['academic_program_name'],
                    'seat_type': row['seat_type'],
                    'year': row['year'],
                    'opening_rank': row['opening_rank'],
                    'closing_rank': row['closing_rank'],
                    'safety_score': self._calculate_safety_score(student_rank, row['closing_rank']),
                    'institute_tier': self._get_institute_tier(row['institute_name'])
                }
                eligible_programs.append(program_info)
        
        # Sort by safety score and institute ranking
        eligible_programs.sort(key=lambda x: (x['safety_score'], 
                                            self.institute_ranking.index(x['institute_name']) 
                                            if x['institute_name'] in self.institute_ranking else 999))
        
        return eligible_programs
    
    def _get_applicable_seat_types(self, category: str) -> List[str]:
        """
        Get all seat types applicable for a given category
        """
        applicable = ['OPEN']  # Everyone can apply for open seats
        
        if category in self.seat_type_mapping:
            applicable.extend(self.seat_type_mapping[category])
        
        return list(set(applicable))  # Remove duplicates
    
    def _calculate_safety_score(self, student_rank: int, closing_rank: int) -> float:
        """
        Calculate safety score (higher is safer)
        """
        if closing_rank <= 0:
            return 0.0
        return max(0, (closing_rank - student_rank) / closing_rank)
    
    def _get_institute_tier(self, institute_name: str) -> str:
        """
        Categorize institutes into tiers based on historical performance
        """
        if institute_name in self.institute_ranking[:7]:  # Top 7 IITs
            return 'Tier 1'
        elif institute_name in self.institute_ranking[:15]:  # Next 8 IITs
            return 'Tier 2'
        else:
            return 'Tier 3'
    
    def get_recommendations(self, student_rank: int, category: str, gender: str, 
                          top_n: int = 20) -> Dict:
        """
        Get top N recommendations with detailed analysis
        
        Returns:
            Dictionary with recommendations, statistics, and insights
        """
        eligible_programs = self.get_eligible_programs(student_rank, category, gender)
        
        # Get top N recommendations
        top_recommendations = eligible_programs[:top_n]
        
        # Generate statistics
        stats = self._generate_statistics(eligible_programs, student_rank)
        
        # Generate insights
        insights = self._generate_insights(eligible_programs, student_rank, category)
        
        return {
            'recommendations': top_recommendations,
            'total_eligible': len(eligible_programs),
            'statistics': stats,
            'insights': insights
        }
    
    def _generate_statistics(self, eligible_programs: List[Dict], student_rank: int) -> Dict:
        """
        Generate statistics about eligible programs
        """
        if not eligible_programs:
            return {}
        
        df = pd.DataFrame(eligible_programs)
        
        return {
            'total_institutes': df['institute_name'].nunique(),
            'total_programs': len(eligible_programs),
            'tier_distribution': df['institute_tier'].value_counts().to_dict(),
            'avg_safety_score': df['safety_score'].mean(),
            'seat_type_distribution': df['seat_type'].value_counts().to_dict()
        }
    
    def _generate_insights(self, eligible_programs: List[Dict], student_rank: int, category: str) -> List[str]:
        """
        Generate insights and recommendations for the student
        """
        insights = []
        
        if not eligible_programs:
            insights.append("No eligible programs found with current criteria. Consider expanding search parameters.")
            return insights
        
        df = pd.DataFrame(eligible_programs)
        
        # Safety insights
        high_safety = df[df['safety_score'] > 0.3]
        if len(high_safety) > 0:
            insights.append(f"You have {len(high_safety)} high-safety options with good admission chances.")
        
        # Tier insights
        tier1_count = len(df[df['institute_tier'] == 'Tier 1'])
        if tier1_count > 0:
            insights.append(f"Great news! You're eligible for {tier1_count} programs in top-tier IITs.")
        
        # Category advantage
        if category != 'OPEN':
            category_programs = df[df['seat_type'] == category]
            if len(category_programs) > 0:
                insights.append(f"Your {category} category provides access to {len(category_programs)} additional programs.")
        
        # Rank position insight
        avg_closing = df['closing_rank'].mean()
        if student_rank < avg_closing * 0.7:
            insights.append("Your rank is well within the safe range for most eligible programs.")
        elif student_rank < avg_closing * 0.9:
            insights.append("Your rank is competitive for most eligible programs.")
        else:
            insights.append("Consider applying to programs with higher safety scores for better chances.")
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Initialize recommender
    recommender = CollegeRecommender()
    
    # Test with sample student profile
    test_rank = 5000
    test_category = "OBC-NCL"
    test_gender = "Male"
    
    print(f"\nTesting recommendations for:")
    print(f"Rank: {test_rank}, Category: {test_category}, Gender: {test_gender}")
    print("=" * 60)
    
    # Get recommendations
    results = recommender.get_recommendations(test_rank, test_category, test_gender, top_n=10)
    
    print(f"\nTotal eligible programs: {results['total_eligible']}")
    print(f"\nTop 10 Recommendations:")
    print("-" * 40)
    
    for i, prog in enumerate(results['recommendations'], 1):
        print(f"{i}. {prog['institute_name']}")
        print(f"   Program: {prog['academic_program_name']}")
        print(f"   Seat Type: {prog['seat_type']}, Safety Score: {prog['safety_score']:.2f}")
        print(f"   Closing Rank: {prog['closing_rank']}, Tier: {prog['institute_tier']}")
        print()
    
    print("\nInsights:")
    for insight in results['insights']:
        print(f"• {insight}")