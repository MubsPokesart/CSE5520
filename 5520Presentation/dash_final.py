import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as stats
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import ttest_ind

# Load and preprocess data (adjust paths as needed)
# Assuming 'seasons_dict' and relevant data structures are already processed in your script.
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
    mean_net_rating = np.mean([team_data['average_net_rating'] for team_data in teams_data])
    mean_offensive_rating = np.mean([team_data['average_offensive_rating'] for team_data in teams_data])
    mean_defensive_rating = np.mean([team_data['average_defensive_rating'] for team_data in teams_data])
    
    for team_data in teams_data:
        team_data['relative_net_rating'] = team_data['average_net_rating'] - mean_net_rating
        team_data['relative_offensive_rating'] = team_data['average_offensive_rating'] - mean_offensive_rating
        team_data['relative_defensive_rating'] = team_data['average_defensive_rating'] - mean_defensive_rating

# Prepare Dash app
app = dash.Dash(__name__)
app.title = "NBA Data Analysis Dashboard"

# Data preprocessing for Plotly
def prepare_violin_data(seasons_dict):
    plot_data = []
    for season, teams in seasons_dict.items():
        for team in teams:
            plot_data.append({
                'Season': season,
                'Conference': team['conference'],
                'Net Rating': team['average_net_rating']
            })
    return pd.DataFrame(plot_data)

def prepare_gmm_data(seasons_dict):
    all_teams_data = []
    for season, teams in seasons_dict.items():
        for team in teams:
            team_data = team.copy()
            team_data['team'] = season + ' ' + team['team']
            all_teams_data.append(team_data)

    df = pd.DataFrame(all_teams_data)
    #Invert defensive rating for better clustering
    df['relative_defensive_rating'] = -df['relative_defensive_rating']
    X = df[['relative_offensive_rating', 'relative_defensive_rating']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    df['Cluster'] = labels
    return df
    
def calculate_p_values(seasons_dict):
    p_values = {}
    for season, teams_data in seasons_dict.items():
        west_data = [team for team in teams_data if team['conference'] == 'Western']
        east_data = [team for team in teams_data if team['conference'] == 'Eastern']
        
        p_values[season] = {
            'Net Rating': stats.ttest_ind(
                [team['average_net_rating'] for team in west_data],
                [team['average_net_rating'] for team in east_data]
            ).pvalue,
            'Offensive Rating': stats.ttest_ind(
                [team['average_offensive_rating'] for team in west_data],
                [team['average_offensive_rating'] for team in east_data]
            ).pvalue,
            'Defensive Rating': stats.ttest_ind(
                [team['average_defensive_rating'] for team in west_data],
                [team['average_defensive_rating'] for team in east_data]
            ).pvalue,
        }
    return pd.DataFrame(p_values).T

# Generate p-value DataFrame
p_value_df = calculate_p_values(seasons_dict)

# Function to generate the Plotly heatmap
def generate_heatmap(p_value_df):
    # Melt DataFrame to long format for Plotly
    heatmap_data = p_value_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='p-value')
    heatmap_data.rename(columns={'index': 'Season'}, inplace=True)

    # Create heatmap using Plotly Express
    fig = px.imshow(
        p_value_df.values,
        labels={"x": "Metric", "y": "Season", "color": "p-value"},
        x=p_value_df.columns,
        y=p_value_df.index,
        color_continuous_scale='YlOrRd_r',
        text_auto='.3f'
    )
    
    fig.update_layout(
        title="Statistical Significance of Conference Differences",
        xaxis_title="Metric",
        yaxis_title="Season",
        coloraxis_colorbar=dict(title="p-value"),
    )
    
    # Annotate heatmap with p-values
    fig.update_traces(texttemplate='%{text}', text=heatmap_data['p-value'].round(3))
    
    return fig

# Data for Dash
violin_data = prepare_violin_data(seasons_dict)
gmm_data = prepare_gmm_data(seasons_dict)

# Layout
app.layout = html.Div([
    html.H1("NBA Data Analysis Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Violin Plots', children=[
            html.H3("Net Ratings by Conference (2009-2019)", style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='season-dropdown',
                options=[{'label': season, 'value': season} for season in violin_data['Season'].unique()],
                value=violin_data['Season'].unique()[0],
                clearable=False
            ),
            dcc.Graph(id='violin-plot')
        ]),
        dcc.Tab(label='GMM Clustering', children=[
            html.H3("Clustering of Teams Based on Ratings", style={'textAlign': 'center'}),
            dcc.Graph(
                id='gmm-clustering',
                figure=px.scatter(
                    gmm_data, x='relative_offensive_rating', y='relative_defensive_rating',
                    color='Cluster', symbol='conference', hover_name='team',
                    labels={
                        'relative_offensive_rating': 'Relative Offensive Rating',
                        'relative_defensive_rating': 'Relative Defensive Rating (Inverted)'
                    },
                    title="GMM Clustering"
                )
            )
        ]),
         dcc.Tab(label='P-Value Heatmap by Season', children=[
            html.H3("P-Value Heatmap", style={'textAlign': 'center'}),
            dcc.Graph(
                id='p-value-heatmap',
                figure=generate_heatmap(p_value_df)  # Initial heatmap
         )])
    ])
])


# Callbacks
@app.callback(
    Output('violin-plot', 'figure'),
    Input('season-dropdown', 'value')
)
def update_violin_plot(selected_season):
    filtered_data = violin_data[violin_data['Season'] == selected_season]
    fig = px.violin(
        filtered_data, x='Conference', y='Net Rating',
        color='Conference', box=True, points='all',
        title=f"Net Ratings by Conference for {selected_season}"
    )
    fig.update_layout(violinmode='group')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)