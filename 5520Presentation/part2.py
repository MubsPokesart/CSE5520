import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests
import networkx as nx
from datetime import datetime

START_DATE = "2009-10-27" # First day filtering
END_DATE = "2019-10-22" # Last day filtering

class TeamObject:

    def __init__(self, name, id, conference):
        self.name = name
        self.id = id
        self.conference = conference
        self.games = {}

# Teams in NBA based on dataset
WESTERN_CONFERENCE_TEAMS = {'Portland Trail Blazers', 'Los Angeles Lakers', 'Dallas Mavericks', 'Golden State Warriors', 'Denver Nuggets',
                            'Los Angeles Clippers', 'San Antonio Spurs', 'Minnesota Timberwolves', 'Memphis Grizzlies', 'New Orleans Hornets',
                            'Phoenix Suns', 'Oklahoma City Thunder', 'Utah Jazz', 'Houston Rockets', 'Sacramento Kings', 'LA Clippers', 'New Orleans Pelicans'}

EASTERN_CONFERENCE_TEAMS = {'Cleveland Cavaliers', 'Atlanta Hawks', 'Miami Heat', 'Boston Celtics',  'Orlando Magic', 'Toronto Raptors',
                            'Chicago Bulls', 'New Jersey Nets', 'Detroit Pistons', 'Charlotte Bobcats', 'Philadelphia 76ers', 'Indiana Pacers', 
                            'Washington Wizards', 'New York Knicks', 'Milwaukee Bucks', 'Brooklyn Nets', 'Charlotte Hornets'}

ALL_NBA_TEAMS = WESTERN_CONFERENCE_TEAMS.union(EASTERN_CONFERENCE_TEAMS)

#METRICS = ['pts_home', 'fg_pct_home', 'fg3_pct_home', 'ft_pct_home', 'reb_home', 'ast_home', 'stl_home', 'blk_home', 'tov_home']

#   Create DataFrame for summary and filter data
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

# Correlation and Regression Analysis
def analyze_correlation_regression(team_objects):
    # Prepare data for analysis
    western_plus_minus = []
    eastern_plus_minus = []
    
    for team in team_objects:
        avg_plus_minus = np.mean(list(team.games.values()))
        if team.conference == 'Western':
            western_plus_minus.append(avg_plus_minus)
        else:
            eastern_plus_minus.append(avg_plus_minus)
    
    # Correlation Analysis
    plt.figure(figsize=(12, 5))
    
    # Plus/Minus Distribution
    plt.subplot(1, 2, 1)
    plt.boxplot([western_plus_minus, eastern_plus_minus], labels=['Western', 'Eastern'])
    plt.title('Plus/Minus Distribution by Conference')
    plt.ylabel('Average Plus/Minus')
    
    # Regression Analysis
    plt.subplot(1, 2, 2)
    x = range(len(western_plus_minus))
    slope_w, intercept_w, r_value_w, _, _ = stats.linregress(x, sorted(western_plus_minus))
    slope_e, intercept_e, r_value_e, _, _ = stats.linregress(x, sorted(eastern_plus_minus))
    
    plt.scatter(x, sorted(western_plus_minus), label=f'Western (R² = {r_value_w**2:.3f})')
    plt.scatter(x, sorted(eastern_plus_minus), label=f'Eastern (R² = {r_value_e**2:.3f})')
    plt.plot(x, [slope_w * xi + intercept_w for xi in x], '--', color='blue')
    plt.plot(x, [slope_e * xi + intercept_e for xi in x], '--', color='orange')
    plt.title('Team Performance Regression')
    plt.xlabel('Team Rank')
    plt.ylabel('Average Plus/Minus')
    plt.legend()
    
    plt.tight_layout()
    return plt

# Network Visualization
def create_network_visualization(team_objects):
    G = nx.Graph()
    
    # Add nodes
    for team in team_objects:
        G.add_node(team.name, conference=team.conference)
    
    # Add edges based on similar performance
    for i, team1 in enumerate(team_objects):
        avg_pm1 = np.mean(list(team1.games.values()))
        for j, team2 in enumerate(team_objects[i+1:], i+1):
            avg_pm2 = np.mean(list(team2.games.values()))
            if abs(avg_pm1 - avg_pm2) < 2:  # Teams with similar performance
                G.add_edge(team1.name, team2.name)
    
    # Create layout
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    node_colors = ['blue' if G.nodes[node]['conference'] == 'Western' else 'red' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Team Performance Similarity Network\nBlue: Western Conference, Red: Eastern Conference')
    plt.axis('off')
    return plt

def perform_clustering_analysis(team_objects):
    # Prepare data for clustering
    team_data = []
    team_labels = []
    team_colors = []
    
    def get_season(date):
        year = date.year
        month = date.month
        
        if year and month < 10:
            return f"{year-1}-{str(year)[-2:]}"
        else:
            return f"{year}-{str(year+1)[-2:]}"
    
    # First, organize games by season for each team
    for team in team_objects:
        # Group games by season
        season_games = {}
        for game_date, plus_minus in team.games.items():
            season = get_season(game_date)
            if season not in season_games:
                season_games[season] = []
            season_games[season].append(plus_minus)
        
        # Calculate metrics for each season
        for season, games in season_games.items():
            total_plus_minus = np.sum(games)
            std_plus_minus = np.std(games)
            
            team_data.append([total_plus_minus, std_plus_minus])
            team_labels.append(f"{season} {team.name}")
            team_colors.append('blue' if team.conference == 'Western' else 'red')
    
    # Convert to numpy array for clustering
    X = np.array(team_data)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create the visualization
    plt.figure(figsize=(20, 12))
    
    # Create scatter plot
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], c=team_colors[i], alpha=0.6, s=100)
        
        # Add team labels with correct season
        plt.annotate(team_labels[i], (X[i, 0], X[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Plot cluster centers
    centers = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers)
    plt.scatter(centers_original[:, 0], centers_original[:, 1], c='yellow', marker='*', s=300,label='Cluster Centers')
    
    # Add labels and title
    plt.xlabel('Total Plus/Minus')
    plt.ylabel('Plus/Minus Standard Deviation')
    plt.title('NBA Team Clustering by Performance (2009-2019)\nBlue: Western Conference, Red: Eastern Conference')
    
    # Add legend
    custom_lines = [plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', label='Western Conference'), plt.Line2D([0], [0], color='red', marker='o', linestyle='None', label='Eastern Conference'), plt.Line2D([0], [0], color='yellow', marker='*', linestyle='None', label='Cluster Centers')]
    plt.legend(handles=custom_lines)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('team_clustering.png', dpi=300, bbox_inches='tight')
    return plt

# Add to the run_all_analyses function
def run_all_analyses(team_objects):
    analyze_correlation_regression(team_objects).savefig('correlation_regression.png')
    create_network_visualization(team_objects).savefig('network_viz.png')
    perform_clustering_analysis(team_objects).savefig('team_clustering.png')

# Run the analysis
run_all_analyses(team_objects)