import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Output, Input, State

external_stylesheets = [dbc.themes.LITERA]
import model

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

title_element = dbc.Container([
    html.H3("Text Simplification model", className="display-3"),
    html.P(
        ["Practical Machine Learning and Deep Learning course project",
         html.Br(),
         html.B("Shamil Hastiev, Qazybek Askarbek and Lev Svalov")],
        className="lead",
        style={"marginLeft": 30}
    ),
    html.Hr(className="my-2")
], style={"marginTop": 30})

text_placeholder = dbc.Textarea(
    placeholder="Input sentence in Russian, max - 200 symbols", maxLength=200, bs_size="lg",
    id="text-placeholder",
    style={"marginTop": 20}
)

button = html.Div([
    dbc.Button(
        "Run",
        size="lg",
        color="primary",
        id="run-button",
        style={"marginTop": 10, }
    ),
    html.Div(id="progress-div", style={"display": "inline-block", "marginLeft": 20, "marginTop": 30})
])


output_div = html.Div([
    html.Div(id="output-1", style={"marginTop": 10}),
    html.Div(id="output-2", style={"marginTop": 10}),
    html.Div(id="output-3", style={"marginTop": 10}),
    html.Div(id="output-4", style={"marginTop": 10}),
    html.Div(id="output-5", style={"marginTop": 10}),
], style={"marginTop": 50, "marginLeft": 30})


@app.callback(
    Output("progress-div", "children"),
    Input("run-button", "n_clicks"),
    State("text-placeholder", "value"),
    prevent_initial_call=True
)
def run_model_state(n_clicks, text):
    if 200 > len(text) > 0:
        return "Starting the model!"
    return "Can not run the model, length of sentence should be from 1 to 200!"


@app.callback(
    Output("output-1", "children"),
    Output("output-2", "children"),
    Output("output-3", "children"),
    Output("output-4", "children"),
    Output("output-5", "children"),
    Input("run-button", "n_clicks"),
    State("text-placeholder", "value"),
    prevent_initial_call=True
)
def get_result(n_clicks, text):
    if 200 > len(text) > 0:
        res = []
        for i, s in enumerate(model.transform_sentence(text)):
            res.append([html.B(str(i + 1)+".  ", style={"display": "inline-block", "marginLeft": 5}), s])
        s1, s2, s3, s4, s5 = tuple(res)
        return s1, s2, s3, s4, s5


children = [title_element, text_placeholder, button, output_div]
app.layout = dbc.Container(children=children)

if __name__ == '__main__':
    app.run_server(debug=True)
