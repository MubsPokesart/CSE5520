from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

def fibonacci(Nterms: int, N1: int, N2: int):
    """Calculates the Fibonacci sequence up to Nterms."""
    Count = 0
    fib_sequence = []

    # Check if the number of terms is valid
    if Nterms <= 0:
        print("Please enter a positive integer")
    else:
        # Generate sequence
        fib_sequence.append(N1)
        while Count < Nterms - 1:
            Nth = N1 + N2
            fib_sequence.append(Nth)
            # update values
            N1 = N2
            N2 = Nth
            Count += 1

    return fib_sequence

if __name__ == "__main__":
    FIBONACCI_NUMBER = 50

    app = Dash(__name__)

    app.layout = html.Div([
        html.H4('Interactive color selection with simple Dash example'),
        html.P("Select color:"),
        dcc.Dropdown(
            id="dropdown",
            options=['Gold', 'MediumTurquoise', 'LightGreen'],
            value='Gold',
            clearable=False,
        ),
        dcc.Graph(id="graph"),
    ])


    @app.callback(
        Output("graph", "figure"), 
        Input("dropdown", "value"))

    def display_color(color):
        fig = go.Figure(
            data=go.Bar(y=fibonacci(FIBONACCI_NUMBER, 0, 1), # replace with your own data source
                        marker_color=color))
        return fig

    app.run_server(debug=True)