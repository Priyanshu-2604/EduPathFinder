import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import argparse
from college_recommender import CollegeRecommender
import warnings
warnings.filterwarnings('ignore')

class PredictionInterface:
    """
    User-friendly interface for IIT college and branch prediction
    Supports both CLI and programmatic access
    """
    
    def __init__(self, model_path: str = 'validated_college_predictor.joblib', 
                 data_path: str = 'advanced_features_iit_ranks.csv'):
        """
        Initialize the prediction interface
        
        Args:
            model_path: Path to the trained model
            data_path: Path to the feature-engineered data
        """
        self.model_path = model_path
        self.data_path = data_path
        
        # Initialize recommender system
        try:
            self.recommender = CollegeRecommender(data_path='features_iit_ranks.csv')
            print("✅ Recommender system initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize recommender system: {e}")
            self.recommender = None
        
        # Load model if available
        try:
            self.model = joblib.load(model_path)
            print("✅ Prediction model loaded successfully")
        except FileNotFoundError:
            print(f"⚠️ Warning: Model file '{model_path}' not found. Please train the model first.")
            self.model = None
        
        # Define valid options
        self.valid_categories = ['OPEN', 'OBC-NCL', 'SC', 'ST', 'EWS']
        self.valid_genders = ['Male', 'Female']
        self.valid_states = ['All States', 'Home State']  # For future home state quota implementation
    
    def validate_input(self, rank: int, category: str, gender: str) -> Tuple[bool, str]:
        """
        Validate user input parameters
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate rank
        if not isinstance(rank, int) or rank <= 0:
            return False, "Rank must be a positive integer"
        
        if rank > 100000:
            return False, "Rank seems too high. Please check your input (max: 100,000)"
        
        # Validate category
        if category not in self.valid_categories:
            return False, f"Category must be one of: {', '.join(self.valid_categories)}"
        
        # Validate gender
        if gender not in self.valid_genders:
            return False, f"Gender must be one of: {', '.join(self.valid_genders)}"
        
        return True, ""
    
    def get_predictions(self, rank: int, category: str, gender: str, 
                       top_n: int = 20, safety_margin: float = 0.1) -> Dict:
        """
        Get comprehensive predictions for a student
        
        Args:
            rank: Student's rank
            category: Student's category
            gender: Student's gender
            top_n: Number of top recommendations to return
            safety_margin: Safety margin for predictions
        
        Returns:
            Dictionary with predictions and analysis
        """
        # Validate input
        is_valid, error_msg = self.validate_input(rank, category, gender)
        if not is_valid:
            return {'error': error_msg}
        
        results = {
            'student_profile': {
                'rank': rank,
                'category': category,
                'gender': gender,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'predictions': {},
            'analysis': {},
            'recommendations': []
        }
        
        # Get recommendations from recommender system
        if self.recommender:
            try:
                recommendations = self.recommender.get_recommendations(
                    rank, category, gender, top_n=top_n
                )
                
                results['predictions'] = {
                    'total_eligible_programs': recommendations['total_eligible'],
                    'top_recommendations': recommendations['recommendations'],
                    'statistics': recommendations['statistics'],
                    'insights': recommendations['insights']
                }
                
                # Generate detailed analysis
                results['analysis'] = self._generate_detailed_analysis(
                    recommendations, rank, category, gender
                )
                
                # Generate actionable recommendations
                results['recommendations'] = self._generate_actionable_recommendations(
                    recommendations, rank, category
                )
                
            except Exception as e:
                results['error'] = f"Error generating predictions: {str(e)}"
        else:
            results['error'] = "Recommender system not available"
        
        return results
    
    def _generate_detailed_analysis(self, recommendations: Dict, rank: int, 
                                  category: str, gender: str) -> Dict:
        """
        Generate detailed analysis of predictions
        """
        analysis = {}
        
        if recommendations['total_eligible'] == 0:
            analysis['overall_assessment'] = "No eligible programs found with current criteria"
            analysis['suggestions'] = [
                "Consider checking if your rank and category are correct",
                "Look into state quota options if available",
                "Consider other engineering entrance exams"
            ]
            return analysis
        
        # Overall assessment
        total_eligible = recommendations['total_eligible']
        if total_eligible > 50:
            analysis['overall_assessment'] = "Excellent! You have many good options"
        elif total_eligible > 20:
            analysis['overall_assessment'] = "Good! You have several viable options"
        elif total_eligible > 5:
            analysis['overall_assessment'] = "Moderate options available"
        else:
            analysis['overall_assessment'] = "Limited options, choose carefully"
        
        # Tier analysis
        if 'statistics' in recommendations and 'tier_distribution' in recommendations['statistics']:
            tier_dist = recommendations['statistics']['tier_distribution']
            analysis['tier_analysis'] = {
                'tier1_programs': tier_dist.get('Tier 1', 0),
                'tier2_programs': tier_dist.get('Tier 2', 0),
                'tier3_programs': tier_dist.get('Tier 3', 0)
            }
        
        # Safety analysis
        top_recs = recommendations.get('recommendations', [])
        if top_recs:
            safety_scores = [rec['safety_score'] for rec in top_recs[:10]]
            avg_safety = np.mean(safety_scores)
            
            if avg_safety > 0.3:
                analysis['safety_assessment'] = "High safety - good chances of admission"
            elif avg_safety > 0.15:
                analysis['safety_assessment'] = "Moderate safety - reasonable chances"
            else:
                analysis['safety_assessment'] = "Low safety - consider backup options"
        
        # Category advantage analysis
        if category != 'OPEN':
            category_programs = len([rec for rec in top_recs if rec['seat_type'] == category])
            open_programs = len([rec for rec in top_recs if rec['seat_type'] == 'OPEN'])
            
            analysis['category_advantage'] = {
                'category_specific_programs': category_programs,
                'open_category_programs': open_programs,
                'advantage_description': f"Your {category} category provides access to {category_programs} additional programs"
            }
        
        return analysis
    
    def _generate_actionable_recommendations(self, recommendations: Dict, 
                                           rank: int, category: str) -> List[str]:
        """
        Generate actionable recommendations for the student
        """
        action_items = []
        
        top_recs = recommendations.get('recommendations', [])
        
        if not top_recs:
            action_items.extend([
                "🔍 Double-check your rank and category information",
                "📞 Contact admission counselors for guidance",
                "🎯 Consider other engineering entrance exams as backup"
            ])
            return action_items
        
        # High safety options
        high_safety = [rec for rec in top_recs if rec['safety_score'] > 0.3]
        if high_safety:
            action_items.append(f"✅ Apply to these {len(high_safety)} high-safety programs: {', '.join([rec['institute_name'] for rec in high_safety[:3]])}")
        
        # Tier 1 opportunities
        tier1_options = [rec for rec in top_recs if rec['institute_tier'] == 'Tier 1']
        if tier1_options:
            action_items.append(f"🎯 Don't miss these top-tier opportunities: {', '.join([rec['institute_name'] for rec in tier1_options[:2]])}")
        
        # Category-specific advice
        if category != 'OPEN':
            category_specific = [rec for rec in top_recs if rec['seat_type'] == category]
            if category_specific:
                action_items.append(f"🏷️ Leverage your {category} quota for these programs: {', '.join([rec['institute_name'] for rec in category_specific[:2]])}")
        
        # Application strategy
        action_items.extend([
            "📝 Prepare all required documents in advance",
            "📅 Keep track of important admission dates and deadlines",
            "💡 Consider multiple rounds of counseling for better options"
        ])
        
        # Backup strategy
        if len(top_recs) < 10:
            action_items.append("🔄 Keep backup options ready in case of changes in cutoffs")
        
        return action_items
    
    def save_prediction_report(self, predictions: Dict, filename: str = None) -> str:
        """
        Save prediction report to a file
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rank = predictions['student_profile']['rank']
            filename = f"prediction_report_rank_{rank}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        return filename
    
    def print_formatted_results(self, predictions: Dict):
        """
        Print formatted prediction results to console
        """
        if 'error' in predictions:
            print(f"❌ Error: {predictions['error']}")
            return
        
        profile = predictions['student_profile']
        print("\n" + "=" * 80)
        print("🎓 IIT COLLEGE & BRANCH PREDICTION REPORT")
        print("=" * 80)
        
        # Student profile
        print(f"\n👤 Student Profile:")
        print(f"   Rank: {profile['rank']:,}")
        print(f"   Category: {profile['category']}")
        print(f"   Gender: {profile['gender']}")
        print(f"   Report Generated: {profile['prediction_date']}")
        
        # Summary
        preds = predictions['predictions']
        print(f"\n📊 Summary:")
        print(f"   Total Eligible Programs: {preds['total_eligible']}")
        
        if 'statistics' in preds:
            stats = preds['statistics']
            print(f"   Eligible Institutes: {stats.get('total_institutes', 'N/A')}")
            if 'tier_distribution' in stats:
                tier_dist = stats['tier_distribution']
                print(f"   Tier 1 Programs: {tier_dist.get('Tier 1', 0)}")
                print(f"   Tier 2 Programs: {tier_dist.get('Tier 2', 0)}")
                print(f"   Tier 3 Programs: {tier_dist.get('Tier 3', 0)}")
        
        # Top recommendations
        print(f"\n🏆 Top Recommendations:")
        print("-" * 80)
        
        top_recs = preds.get('top_recommendations', [])
        for i, rec in enumerate(top_recs[:10], 1):
            print(f"{i:2d}. {rec['institute_name']}")
            print(f"    📚 Program: {rec['academic_program_name']}")
            print(f"    🎫 Seat Type: {rec['seat_type']} | 🏅 Tier: {rec['institute_tier']}")
            print(f"    📈 Safety Score: {rec['safety_score']:.2f} | 🔢 Closing Rank: {rec['closing_rank']:,}")
            print()
        
        # Analysis
        if 'analysis' in predictions:
            analysis = predictions['analysis']
            print(f"\n🔍 Analysis:")
            print(f"   Overall Assessment: {analysis.get('overall_assessment', 'N/A')}")
            if 'safety_assessment' in analysis:
                print(f"   Safety Assessment: {analysis['safety_assessment']}")
        
        # Insights
        if 'insights' in preds:
            print(f"\n💡 Key Insights:")
            for insight in preds['insights']:
                print(f"   • {insight}")
        
        # Recommendations
        if 'recommendations' in predictions:
            print(f"\n🎯 Action Items:")
            for rec in predictions['recommendations']:
                print(f"   {rec}")
        
        print("\n" + "=" * 80)
    
    def run_interactive_session(self):
        """
        Run interactive CLI session for predictions
        """
        print("\n🎓 Welcome to IIT College & Branch Predictor!")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                print("\nPlease enter your details:")
                
                rank = int(input("Enter your rank: "))
                
                print(f"\nAvailable categories: {', '.join(self.valid_categories)}")
                category = input("Enter your category: ").strip().upper()
                
                print(f"\nAvailable genders: {', '.join(self.valid_genders)}")
                gender = input("Enter your gender: ").strip().title()
                
                # Get predictions
                print("\n🔄 Generating predictions...")
                predictions = self.get_predictions(rank, category, gender)
                
                # Display results
                self.print_formatted_results(predictions)
                
                # Ask if user wants to save report
                save_report = input("\nDo you want to save this report? (y/n): ").strip().lower()
                if save_report == 'y':
                    filename = self.save_prediction_report(predictions)
                    print(f"✅ Report saved as: {filename}")
                
                # Ask if user wants to continue
                continue_session = input("\nDo you want to make another prediction? (y/n): ").strip().lower()
                if continue_session != 'y':
                    break
                    
            except ValueError:
                print("❌ Invalid input. Please enter a valid number for rank.")
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using IIT College Predictor!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
        
        print("\n👋 Thank you for using IIT College Predictor!")

def main():
    """
    Main function for CLI usage
    """
    parser = argparse.ArgumentParser(description='IIT College & Branch Predictor')
    parser.add_argument('--rank', type=int, help='Student rank')
    parser.add_argument('--category', type=str, help='Student category (OPEN, OBC-NCL, SC, ST, EWS)')
    parser.add_argument('--gender', type=str, help='Student gender (Male, Female)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--save', action='store_true', help='Save prediction report')
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = PredictionInterface()
    
    if args.interactive or not all([args.rank, args.category, args.gender]):
        # Run interactive session
        interface.run_interactive_session()
    else:
        # Run with provided arguments
        predictions = interface.get_predictions(args.rank, args.category, args.gender)
        interface.print_formatted_results(predictions)
        
        if args.save:
            filename = interface.save_prediction_report(predictions)
            print(f"\n✅ Report saved as: {filename}")

if __name__ == "__main__":
    main()