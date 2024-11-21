import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from typing import List, Dict, Tuple

START_DATE = "2009-10-27" # First day filtering
END_DATE = "2019-10-22" # Last day filtering

class TeamObject:
    def __init__(self, name, id, conference):
        self.name = name
        self.id = id
        self.conference = conference
        self.games = {}

# Teams in NBA based on dataset
WESTERN_CONFERENCE_TEAMS = {'Portland Trail Blazers', 'Los Angeles Lakers', 'Dallas Mavericks', 'Golden State Warriors', 'Denver Nuggets', 'Los Angeles Clippers', 'San Antonio Spurs', 'Minnesota Timberwolves', 'Memphis Grizzlies', 'New Orleans Hornets', 'Phoenix Suns', 'Oklahoma City Thunder', 'Utah Jazz', 'Houston Rockets', 'Sacramento Kings', 'LA Clippers', 'New Orleans Pelicans'}
EASTERN_CONFERENCE_TEAMS = {'Cleveland Cavaliers', 'Atlanta Hawks', 'Miami Heat', 'Boston Celtics',  'Orlando Magic', 'Toronto Raptors', 'Chicago Bulls', 'New Jersey Nets', 'Detroit Pistons', 'Charlotte Bobcats', 'Philadelphia 76ers', 'Indiana Pacers', 'Washington Wizards', 'New York Knicks', 'Milwaukee Bucks', 'Brooklyn Nets', 'Charlotte Hornets'}
ALL_NBA_TEAMS = WESTERN_CONFERENCE_TEAMS.union(EASTERN_CONFERENCE_TEAMS)

# Create DataFrame for summary and filter data
summary_df = pd.read_csv("game.csv")
summary_df['game_date'] = pd.to_datetime(summary_df['game_date'])
range_dataframe = summary_df[(summary_df['game_date'] >= START_DATE) & (summary_df['game_date'] < END_DATE)]

# Find every unique team in the dataset and pass them to the TeamObject with their name and id
teams = range_dataframe['team_id_home'].unique()
team_objects = []
for team in teams:
    team_name = range_dataframe[range_dataframe['team_id_home'] == team]['team_name_home'].iloc[0]
    if team_name in WESTERN_CONFERENCE_TEAMS:
        team_objects.append(TeamObject(team_name, team, 'Western'))
    elif team_name in EASTERN_CONFERENCE_TEAMS:
        team_objects.append(TeamObject(team_name, team, 'Eastern'))

# Iterate through every team and find their games
for team in team_objects:
    team_games = range_dataframe[(range_dataframe['team_id_home'] == team.id) | (range_dataframe['team_id_away'] == team.id)]
    # Iterate through every game and find the metrics for plus/minus data, plus_minus_home is for team_id_home and plus_minus_away is for team_id_away
    for index, game in team_games.iterrows():
        if game['team_id_home'] == team.id:
            plus_minus = game['plus_minus_home']
            team.games[game['game_date']] = plus_minus
        elif game['team_id_away'] == team.id:
            plus_minus = game['plus_minus_away']
            team.games[game['game_date']] = plus_minus

# Define metrics (keep commented out original list)
METRICS = ['plus_minus_home', 'plus_minus_away', 'pts_home', 'pts_away', 'fg_pct_home', 'fg_pct_away', 'fg3_pct_home', 'fg3_pct_away', 'ft_pct_home', 'ft_pct_away', 'reb_home', 'reb_away', 'ast_home', 'ast_away', 'stl_home', 'stl_away', 'blk_home', 'blk_away', 'tov_home', 'tov_away']

class EnhancedTeamAnalytics:
    def __init__(self, name: str, id: int, conference: str):
        self.name = name
        self.id = id
        self.conference = conference
        self.games = {}
        self.rolling_plus_minus = {}
        self.adjusted_plus_minus = {}
        self.strength_of_schedule = 0.0
        self.home_court_advantage = 0.0
        
    def calculate_advanced_metrics(self, all_teams: List['EnhancedTeamAnalytics'], 
                                 range_dataframe: pd.DataFrame) -> None:
        """
        Calculate advanced plus/minus metrics with refined adjustments
        """
        # Calculate rolling plus/minus (10-game window)
        dates = sorted(self.games.keys())
        values = [self.games[date] for date in dates]
        rolling_avg = pd.Series(values).rolling(window=10, min_periods=1).mean()
        self.rolling_plus_minus = dict(zip(dates, rolling_avg))
        
        # Calculate home court advantage (scaled down)
        home_games = range_dataframe[range_dataframe['team_id_home'] == self.id]
        away_games = range_dataframe[range_dataframe['team_id_away'] == self.id]
        self.home_court_advantage = (
            (home_games['plus_minus_home'].mean() - 
             away_games['plus_minus_away'].mean()) * 0.5  # Scale factor to reduce impact
        )
        
        # Calculate strength of schedule with normalization
        all_team_ratings = []
        for team in all_teams:
            if team.games:
                team_rating = np.mean(list(team.games.values()))
                all_team_ratings.append(team_rating)
        
        league_avg = np.mean(all_team_ratings) if all_team_ratings else 0
        league_std = np.std(all_team_ratings) if all_team_ratings else 1
        
        opponent_ratings = []
        for date, plus_minus in self.games.items():
            game = range_dataframe[range_dataframe['game_date'] == date]
            opponent_id = (
                game['team_id_away'].iloc[0] 
                if game['team_id_home'].iloc[0] == self.id 
                else game['team_id_home'].iloc[0]
            )
            
            opponent = next((team for team in all_teams if team.id == opponent_id), None)
            if opponent and opponent.games:
                opponent_rating = np.mean(list(opponent.games.values()))
                # Normalize the rating
                opponent_rating = (opponent_rating - league_avg) / league_std if league_std != 0 else 0
                opponent_ratings.append(opponent_rating)
        
        self.strength_of_schedule = np.mean(opponent_ratings) if opponent_ratings else 0
        
        # Calculate adjusted plus/minus with scaled adjustments
        GARBAGE_TIME_FACTOR = 0.75  # Less aggressive garbage time adjustment
        SOS_FACTOR = 0.3  # Reduced strength of schedule impact
        HCA_FACTOR = 0.5  # Reduced home court advantage impact
        
        for date, raw_plus_minus in self.games.items():
            game = range_dataframe[range_dataframe['game_date'] == date]
            is_home = game['team_id_home'].iloc[0] == self.id
            
            # Home court adjustment (scaled)
            hca_adjustment = (self.home_court_advantage * HCA_FACTOR) if is_home else (-self.home_court_advantage * HCA_FACTOR)
            
            # Strength of schedule adjustment (scaled)
            sos_adjustment = self.strength_of_schedule * SOS_FACTOR
            
            # Garbage time adjustment (less aggressive)
            garbage_time = self._is_garbage_time(game)
            base_value = raw_plus_minus * (GARBAGE_TIME_FACTOR if garbage_time else 1.0)
            
            # Final adjusted value
            self.adjusted_plus_minus[date] = base_value + hca_adjustment + sos_adjustment
    
    def _is_garbage_time(self, game: pd.Series) -> bool:
        """
        More nuanced garbage time detection
        """
        GARBAGE_TIME_THRESHOLD = 25  # Increased threshold
        point_diff = abs(game['pts_home'].iloc[0] - game['pts_away'].iloc[0])
        return point_diff >= GARBAGE_TIME_THRESHOLD
    
    def get_team_efficiency(self) -> Dict[str, float]:
        """Calculate team efficiency metrics"""
        metrics = {
            'raw_plus_minus_avg': np.mean(list(self.games.values())),
            'adjusted_plus_minus_avg': np.mean(list(self.adjusted_plus_minus.values())),
            'rolling_plus_minus_last': list(self.rolling_plus_minus.values())[-1],
            'home_court_impact': self.home_court_advantage,
            'schedule_strength': self.strength_of_schedule,
            'team_name': self.name,
            'conference': self.conference
        }
        # Round all numeric values to 3 decimal places
        return {k: round(v, 3) if isinstance(v, (int, float)) else v 
                for k, v in metrics.items()}

def analyze_league_plus_minus(range_dataframe: pd.DataFrame, 
                            team_objects: List[object]) -> pd.DataFrame:
    """
    Analyze plus/minus statistics across the league
    """
    enhanced_teams = [
        EnhancedTeamAnalytics(team.name, team.id, team.conference) 
        for team in team_objects
    ]
    
    for team in enhanced_teams:
        original = next(t for t in team_objects if t.id == team.id)
        team.games = original.games.copy()
    
    for team in enhanced_teams:
        team.calculate_advanced_metrics(enhanced_teams, range_dataframe)
    
    results = [team.get_team_efficiency() for team in enhanced_teams]
    df = pd.DataFrame(results)
    
    # Sort by adjusted plus/minus for better visualization
    return df.sort_values('adjusted_plus_minus_avg', ascending=False)

print(analyze_league_plus_minus(range_dataframe, team_objects))

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests
import networkx as nx
from datetime import datetime