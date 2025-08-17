import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class IITDataAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        
    def get_course_analysis(self, course_name: str = None) -> pd.DataFrame:
        """Analyze ranks for a specific course across all IITs"""
        if course_name:
            return self.df[self.df['course_name'].str.contains(course_name, case=False, na=False)]
        return self.df
    
    def get_institute_analysis(self, institute_name: str) -> pd.DataFrame:
        """Analyze all courses for a specific IIT"""
        return self.df[self.df['institute_name'].str.contains(institute_name, case=False, na=False)]
    
    def predict_admission_chances(self, rank: int, category: str = "General") -> pd.DataFrame:
        """Predict which courses/IITs a student can get based on rank"""
        filtered_df = self.df[
            (self.df['category'] == category) & 
            (self.df['closing_rank'] >= rank) &
            (self.df['closing_rank'].notna())
        ]
        
        return filtered_df.sort_values('closing_rank')[
            ['institute_name', 'course_name', 'opening_rank', 'closing_rank']
        ]
    
    def plot_rank_trends(self, course_name: str):
        """Plot rank trends for a course over years"""
        course_data = self.get_course_analysis(course_name)
        
        if course_data.empty:
            print(f"No data found for course: {course_name}")
            return
        
        plt.figure(figsize=(12, 8))
        
        for category in course_data['category'].unique():
            cat_data = course_data[course_data['category'] == category]
            plt.plot(cat_data['year'], cat_data['closing_rank'], 
                    marker='o', label=f'{category} - Closing Rank')
        
        plt.title(f'Rank Trends for {course_name}')
        plt.xlabel('Year')
        plt.ylabel('Rank')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def create_rank_comparison_chart(self):
        """Create a comparison chart of closing ranks across top courses"""
        top_courses = self.df.groupby('course_name')['closing_rank'].min().sort_values().head(10)
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=self.df[self.df['course_name'].isin(top_courses.index)], 
                   x='course_name', y='closing_rank')
        plt.xticks(rotation=45, ha='right')
        plt.title('Closing Rank Distribution for Top Courses')
        plt.tight_layout()
        plt.show()
