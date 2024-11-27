import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import scipy.stats as stats
from typing import List, Dict, Tuple

START_DATE = "2009-10-27" # First day filtering
END_DATE = "2019-10-22" # Last day filtering


class TeamObject:
    # Static season dates dictionary
    SEASON_DATES = {
        "2009-10": {"start": "2009-10-27", "end": "2010-06-17"},
        "2010-11": {"start": "2010-10-26", "end": "2011-06-12"},
        "2011-12": {"start": "2011-12-25", "end": "2012-06-21"},
        "2012-13": {"start": "2012-10-30", "end": "2013-06-20"},
        "2013-14": {"start": "2013-10-29", "end": "2014-06-15"},
        "2014-15": {"start": "2014-10-28", "end": "2015-06-16"},
        "2015-16": {"start": "2015-10-27", "end": "2016-06-19"},
        "2016-17": {"start": "2016-10-25", "end": "2017-06-12"},
        "2017-18": {"start": "2017-10-17", "end": "2018-06-08"},
        "2018-19": {"start": "2018-10-16", "end": "2019-06-13"}
    }

    def __init__(self, name, id, conference):
        self.name = name
        self.id = id
        self.conference = conference
        self.games = {}

    @staticmethod
    def convert_to_datetime(date_input):
        """
        Convert various date formats to datetime object
        """
        if isinstance(date_input, pd.Timestamp):
            return date_input.to_pydatetime()
        elif isinstance(date_input, str):
            return datetime.strptime(date_input, "%Y-%m-%d")
        elif isinstance(date_input, datetime):
            return date_input
        else:
            raise ValueError(f"Unsupported date format: {type(date_input)}")

    @classmethod
    def get_season_dates(cls, season):
        """
        Get the start and end dates for a specific season
        Args:
            season (str): Season in format "YYYY-YY"
        Returns:
            dict: Dictionary containing start and end dates for the season
        """
        return cls.SEASON_DATES.get(season)

    @classmethod
    def is_date_in_season(cls, date_input, season):
        """
        Check if a given date falls within a specific season
        Args:
            date_input: Date in string, Timestamp, or datetime format
            season (str): Season in format "YYYY-YY"
        Returns:
            bool: True if date is within season, False otherwise
        """
        season_dates = cls.get_season_dates(season)
        if not season_dates:
            return False
        
        date = cls.convert_to_datetime(date_input)
        season_start = datetime.strptime(season_dates["start"], "%Y-%m-%d")
        season_end = datetime.strptime(season_dates["end"], "%Y-%m-%d")
        
        return season_start <= date <= season_end

    def get_games_by_season(self):
        """
        Organizes team's games by season.
        Returns:
            list: List of dictionaries where each dictionary contains games for a season
                  with keys formatted as 'season_teamname'
        """
        seasons_games = []
        
        for season in self.SEASON_DATES:
            season_dict = {f"{season} {self.name}": []}
            
            # Go through all games and check if they belong to this season
            for game_date, game_data in self.games.items():
                if self.is_date_in_season(game_date, season):
                    # Convert Timestamp to string format if needed
                    date_str = game_date
                    if isinstance(game_date, pd.Timestamp):
                        date_str = game_date.strftime("%Y-%m-%d")
                    
                    season_dict[f"{season} {self.name}"].append({
                        "date": date_str,
                        **game_data  # Include all other game data
                    })
            
            # Only append if there are games in this season
            if season_dict[f"{season} {self.name}"]:
                seasons_games.append(season_dict)
        
        return seasons_games

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

# Generate possession-based metrics for each game
for team in team_objects:
    team_games = range_dataframe[(range_dataframe['team_id_home'] == team.id) | (range_dataframe['team_id_away'] == team.id)]

    # Iterate through every game and find the metrics
    for index, game in team_games.iterrows():
        game_stats = {}
        
        if game['team_id_home'] == team.id:
            # Team is home team
            # Calculate possession for team (home perspective)
            team_orb_denom = np.maximum(game['oreb_home'] + game['dreb_away'], 1)  # Avoid division by zero
            opp_orb_denom = np.maximum(game['oreb_away'] + game['dreb_home'], 1)
            
            team_orb_pct = np.divide(game['oreb_home'], team_orb_denom)
            opp_orb_pct = np.divide(game['oreb_away'], opp_orb_denom)
            
            # Vectorized possession calculations
            team_poss = np.sum([
                game['fga_home'],
                0.4 * game['fta_home'],
                -1.07 * team_orb_pct * (game['fga_home'] - game['fgm_home']),
                game['tov_home']
            ])
            
            opp_poss = np.sum([
                game['fga_away'],
                0.4 * game['fta_away'],
                -1.07 * opp_orb_pct * (game['fga_away'] - game['fgm_away']),
                game['tov_away']
            ])
            
            # Store stats using numpy operations
            game_stats['plus_minus'] = game['plus_minus_home']
            game_stats['offensive_rating'] = np.multiply(np.divide(game['pts_home'], team_poss), 100)
            game_stats['defensive_rating'] = np.multiply(np.divide(game['pts_away'], opp_poss), 100)
            
        elif game['team_id_away'] == team.id:
            # Team is away team
            # Calculate possession for team (away perspective)
            team_orb_denom = np.maximum(game['oreb_away'] + game['dreb_home'], 1)
            opp_orb_denom = np.maximum(game['oreb_home'] + game['dreb_away'], 1)
            
            team_orb_pct = np.divide(game['oreb_away'], team_orb_denom)
            opp_orb_pct = np.divide(game['oreb_home'], opp_orb_denom)
            
            # Vectorized possession calculations
            team_poss = np.sum([
                game['fga_away'],
                0.4 * game['fta_away'],
                -1.07 * team_orb_pct * (game['fga_away'] - game['fgm_away']),
                game['tov_away']
            ])
            
            opp_poss = np.sum([
                game['fga_home'],
                0.4 * game['fta_home'],
                -1.07 * opp_orb_pct * (game['fga_home'] - game['fgm_home']),
                game['tov_home']
            ])
            
            # Store stats using numpy operations
            game_stats['plus_minus'] = game['plus_minus_away']
            game_stats['offensive_rating'] = np.multiply(np.divide(game['pts_away'], team_poss), 100)
            game_stats['defensive_rating'] = np.multiply(np.divide(game['pts_home'], opp_poss), 100)
        
        # Add net rating
        game_stats['net_rating'] = game_stats['offensive_rating'] - game_stats['defensive_rating']
        game_stats['possessions'] = np.mean([team_poss, opp_poss])
        
        # Store all stats for this game date
        team.games[game['game_date']] = game_stats

# Generate possession-based metrics for every team in each season
seasons_dict = {season: [] for season in TeamObject.SEASON_DATES.keys()}
season_name_length = len(list(TeamObject.SEASON_DATES.keys())[0])

for team in team_objects:
    team_seasons = team.get_games_by_season()

    for season in team_seasons:
        season_name = list(season.keys())[0]
        game_statistics = list(season.values())[0]
        average_net_rating = np.mean([game['net_rating'] for game in game_statistics])
        average_offensive_rating = np.mean([game['offensive_rating'] for game in game_statistics])
        average_defensive_rating = np.mean([game['defensive_rating'] for game in game_statistics])
        average_plus_minus = np.mean([game['plus_minus'] for game in game_statistics])

        seasons_dict[season_name[:season_name_length]].append({
            "team": team.name,
            "conference": team.conference,
            "average_offensive_rating": average_offensive_rating,
            "average_defensive_rating": average_defensive_rating,
            "average_net_rating": average_net_rating,
            "average_plus_minus": average_plus_minus
        })

# Add relative net rating to each season
for season, teams_data in seasons_dict.items():
    net_ratings = [team_data['average_net_rating'] for team_data in teams_data]
    mean_net_rating = np.mean(net_ratings)
    
    for team_data in teams_data:
        team_data['relative_net_rating'] = team_data['average_net_rating'] - mean_net_rating

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats

# Prepare the data
seasons_data = {
    '2009-10': [team for team in seasons_dict['2009-10']],
    '2010-11': [team for team in seasons_dict['2010-11']],
    '2011-12': [team for team in seasons_dict['2011-12']],
    '2012-13': [team for team in seasons_dict['2012-13']],
    '2013-14': [team for team in seasons_dict['2013-14']],
    '2014-15': [team for team in seasons_dict['2014-15']],
    '2015-16': [team for team in seasons_dict['2015-16']],
    '2016-17': [team for team in seasons_dict['2016-17']],
    '2017-18': [team for team in seasons_dict['2017-18']],
    '2018-19': [team for team in seasons_dict['2018-19']]
}

# 1. Conference-level Statistical Analysis
def conference_statistical_analysis(metric):
    """
    Perform statistical analysis comparing Eastern and Western conferences
    
    Args:
    metric (str): The performance metric to analyze
    
    Returns:
    dict: Statistical test results
    """
    results = {}
    
    for season, teams in seasons_data.items():
        # Separate Eastern and Western conference data
        eastern_teams = [team[metric] for team in teams if team['conference'] == 'Eastern']
        western_teams = [team[metric] for team in teams if team['conference'] == 'Western']
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(western_teams, eastern_teams)
        
        results[season] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'western_mean': np.mean(western_teams),
            'eastern_mean': np.mean(eastern_teams)
        }
    
    return results

# Perform analysis for different metrics
offensive_rating_analysis = conference_statistical_analysis('average_offensive_rating')
defensive_rating_analysis = conference_statistical_analysis('average_defensive_rating')
net_rating_analysis = conference_statistical_analysis('average_net_rating')

# 2. ARIMA Time Series Analysis
def prepare_conference_timeseries(metric):
    """
    Prepare time series data for ARIMA analysis
    
    Args:
    metric (str): The performance metric to analyze
    
    Returns:
    tuple: Western and Eastern conference time series
    """
    western_series = [np.mean([team[metric] for team in seasons_data[season] if team['conference'] == 'Western']) 
                      for season in sorted(seasons_data.keys())]
    eastern_series = [np.mean([team[metric] for team in seasons_data[season] if team['conference'] == 'Eastern']) 
                      for season in sorted(seasons_data.keys())]
    
    return western_series, eastern_series

def arima_analysis(series):
    """
    Perform ARIMA analysis on a time series
    
    Args:
    series (list): Time series data
    
    Returns:
    dict: ARIMA model results
    """
    # Fit ARIMA model
    model = ARIMA(series, order=(1,1,1))
    results = model.fit()
    
    return {
        'coefficients': results.params,
        'p_values': results.pvalues,
        'trend': results.forecast(steps=1)[0]
    }

# Perform ARIMA analysis for different metrics
offensive_rating_arima = {
    'Western': arima_analysis(prepare_conference_timeseries('average_offensive_rating')[0]),
    'Eastern': arima_analysis(prepare_conference_timeseries('average_offensive_rating')[1])
}

net_rating_arima = {
    'Western': arima_analysis(prepare_conference_timeseries('average_net_rating')[0]),
    'Eastern': arima_analysis(prepare_conference_timeseries('average_net_rating')[1])
}

# 3. K-means Clustering
def perform_kmeans_clustering(metric):
    """
    Perform K-means clustering on team performance
    
    Args:
    metric (str): The performance metric to analyze
    
    Returns:
    dict: Clustering results
    """
    # Prepare data for clustering
    cluster_data = []
    team_names = []
    conferences = []
    
    for season in seasons_data.keys():
        for team in seasons_data[season]:
            cluster_data.append([
                team['average_offensive_rating'],
                team['average_defensive_rating'],
                team['average_net_rating']
            ])
            team_names.append(f"{season} {team['team']}")
            conferences.append(team['conference'])
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Analyze cluster composition
    cluster_composition = {}
    for i in range(4):
        cluster_composition[i] = {
            'total_teams': sum(clusters == i),
            'western_teams': sum((clusters == i) & (np.array(conferences) == 'Western')),
            'eastern_teams': sum((clusters == i) & (np.array(conferences) == 'Eastern')),
            'cluster_center': kmeans.cluster_centers_[i]
        }
    
    return {
        'clusters': clusters,
        'team_names': team_names,
        'conferences': conferences,
        'cluster_composition': cluster_composition
    }

# Perform K-means clustering
kmeans_results = perform_kmeans_clustering('average_net_rating')

# Visualization and Summary
def generate_summary():
    """
    Generate a comprehensive summary of the analysis
    """
    summary = "NBA Conference Performance Analysis (2009-2019)\n\n"
    
    # Statistical Significance Summary
    summary += "1. Statistical Significance Analysis:\n"
    for metric, analysis in [
        ('Offensive Rating', offensive_rating_analysis),
        ('Defensive Rating', defensive_rating_analysis),
        ('Net Rating', net_rating_analysis)
    ]:
        summary += f"   {metric} Comparison:\n"
        significant_seasons = [
            season for season, result in analysis.items() 
            if result['p_value'] < 0.05
        ]
        summary += f"   - Statistically significant seasons: {significant_seasons}\n"
        summary += f"   - Seasons with Western Conference advantage:\n"
        for season, result in analysis.items():
            if result['western_mean'] > result['eastern_mean']:
                summary += f"     * {season}: West mean = {result['western_mean']:.2f}, East mean = {result['eastern_mean']:.2f}\n"
    
    # ARIMA Trend Summary
    summary += "\n2. ARIMA Time Series Trends:\n"
    for metric, arima_results in [
        ('Offensive Rating', offensive_rating_arima),
        ('Net Rating', net_rating_arima)
    ]:
        summary += f"   {metric} Trend:\n"
        for conference, results in arima_results.items():
            summary += f"   - {conference} Conference:\n"
            summary += f"     * Forecast trend: {results['trend']:.2f}\n"
    
    # K-means Clustering Summary
    summary += "\n3. K-means Clustering Insights:\n"
    for cluster, composition in kmeans_results['cluster_composition'].items():
        summary += f"   Cluster {cluster}:\n"
        summary += f"   - Total Teams: {composition['total_teams']}\n"
        summary += f"   - Western Teams: {composition['western_teams']}\n"
        summary += f"   - Eastern Teams: {composition['eastern_teams']}\n"
    
    return summary

# Generate and print the summary
analysis_summary = generate_summary()
print(analysis_summary)

def create_visualizations(seasons_data, kmeans_results):
    """
    Create visualizations to support the analysis
    
    Args:
    seasons_data (dict): Dictionary of seasons and their team data
    kmeans_results (dict): Results from K-means clustering
    """
    # Conference Comparison Boxplot
    plt.figure(figsize=(15,8))
    conference_data = []
    labels = []
    
    for season in sorted(seasons_data.keys()):
        western_ratings = [team['average_net_rating'] for team in seasons_data[season] if team['conference'] == 'Western']
        eastern_ratings = [team['average_net_rating'] for team in seasons_data[season] if team['conference'] == 'Eastern']
        
        conference_data.extend([western_ratings, eastern_ratings])
        labels.extend([f'{season} West', f'{season} East'])
    
    plt.boxplot(conference_data, tick_labels=labels)
    plt.title('Net Rating by Conference and Season', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Net Rating')
    plt.tight_layout()
    plt.show()
    
    # Scatter plot for K-means Clustering
    plt.figure(figsize=(12,8))
    
    # Prepare data for scatter plot
    cluster_data = np.array([
        [team['average_offensive_rating'], 
         team['average_defensive_rating'], 
         team['average_net_rating']] 
        for season in seasons_data.values() 
        for team in season
    ])
    
    # Use StandardScaler if you want to normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Perform K-means again to get consistent results
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Create scatter plot
    scatter = plt.scatter(
        scaled_data[:, 0],  # Offensive Rating
        scaled_data[:, 1],  # Defensive Rating
        c=clusters,
        cmap='viridis',
        alpha=0.7
    )
    
    plt.title('Team Performance Clusters', fontsize=16)
    plt.xlabel('Standardized Offensive Rating', fontsize=12)
    plt.ylabel('Standardized Defensive Rating', fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()
    
    # Additional: Conference Performance Over Time
    plt.figure(figsize=(15,8))
    
    # Prepare data for line plot
    western_net_ratings = [
        np.mean([team['average_net_rating'] for team in seasons_data[season] if team['conference'] == 'Western'])
        for season in sorted(seasons_data.keys())
    ]
    eastern_net_ratings = [
        np.mean([team['average_net_rating'] for team in seasons_data[season] if team['conference'] == 'Eastern'])
        for season in sorted(seasons_data.keys())
    ]
    
    plt.plot(sorted(seasons_data.keys()), western_net_ratings, marker='o', label='Western Conference')
    plt.plot(sorted(seasons_data.keys()), eastern_net_ratings, marker='o', label='Eastern Conference')
    
    plt.title('Conference Net Rating Trends', fontsize=16)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Average Net Rating', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage (commented out as it requires the original context)
create_visualizations(seasons_data, kmeans_results)

print("Visualization function updated and ready to use.")
