""" Program which allows the user to enter a citation context through Dash, and gets recommendations using a hybrid of 3 models """
# For dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
#import plotly.graph_objs as go

import pandas as pd
import numpy as np

# For hyperdoc2vec
from HyperDoc2Vec import *
import requests

# For clean text
import re
import contractions
from gensim.parsing import preprocessing

# For hybrid
from numpy.random import choice
from copy import deepcopy
from collections import Counter

# For database retrieval
import psycopg2
import psycopg2.extras
conn = psycopg2.connect("dbname=MAG19 user=mag password=1maG$ host=shetland.informatik.uni-freiburg.de")
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

from get_recommendations import clean_text, hd2v_wvindvout_recommend, solr_cited_recommend,\
 solr_recommend, search_solr_parse_json, hybrid_recommend, get_paper_details, get_topn

# Hybrid global calculations: set hybrid_weights (same for all recommendations)

# Set topn
topn = 500
hd2vmodel = HyperDoc2Vec.load('/home/ashwath/Programs/MAGCS/AllYearsFiles/models/magcsenglish_window20_all.model')

hd2vreducedmodel = HyperDoc2Vec.load('/home/ashwath/Programs/MAGCS/AllYearsFiles/models/magcsenglish_window20_mincitations50_all.model')

external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
#app.css.append_css({'external_url': 'https://rawgit.com/lwileczek/Dash/master/undo_redo5.css'})

theme =  {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}
app.title = 'HybridCite'
app.layout = html.Div(id='main',
    #style={'height':'100vh', 'width': '100%'},
    #style={'background-image': 'url("/assets/trianglifyPuBuGn.png")', 'height':'100vh'},
    children=[
    html.H1(children='HybridCite', style={'textAlign': 'center'}), 
    #daq.ToggleSwitch(
    #    id='daq-light-dark-theme',
    #    label=['Light', 'Dark'],
    #    style={'width': '250px', 'margin': 'auto', 'float':'right'}, 
    #    value=False,
    #),
    html.P(children="""Enter a citation context (a couple of sentences or so of text -- max 500 characters) and get back citation recommendations.
                        The recommendations are from a semi-genetic hybrid model based on Hyperdoc2vec and BM25.
                        The recommender can be set to recommend papers with over 50 citations only (this uses a different model),
                        it can also be set to exclude papers with over 500 citations. By default, it recommends papers with any 
                        number of citations. The recommended papers are linked to their corresponding Microsoft Academic pages.
                        You can also hover on a recommendation to see its abstract, year and number of citations.""",
            style={
                'textAlign': 'justify',
                #'color': colours['text'],
                'fontSize': '1.2em',
                'width': '90%',
                'margin-left': '1%'
            }
    ),
    html.Br(),

    dcc.Textarea(id='context-state', placeholder='Enter a citation context', value='', 
        style={'width': '90%', 'height': '40%', 'margin-left': '1%','margin-bottom': '1%',
         'fontSize': '1.2em'}, maxLength=500),
    html.Br(),
    #html.Div(children=[
        html.Label(id='limitslabel',
               style={
                'textAlign': 'left',
                'fontSize': '1.2em',
                'margin-left': '1%',
                'margin-bottom': '1%'
            }, children='Set the lower and upper limits for the number of citations of recommended papers.'),
        html.Br(),
        daq.ToggleSwitch(
            id='lowerlimit',
            label=['No lower limit', '50 citations'],
            style={'width': '50%', 'margin-left': '1%', 'float': 'left'}, 
            value=False
            ),
        html.Br(),
        daq.ToggleSwitch(
            id='upperlimit',
            label=['No upper limit', '500 citations'],
            style={'width': '50%', 'margin-left': '1%', 'float': 'left', 'margin-top': '1%'}, 
            value=False
            ),
        html.Div(children=[

        daq.Slider(
            id='numrecommendations',
            min=10,
            max=50,
            value=10,
            handleLabel={"showCurrentValue": True,"label": "Recommendations:"},
            step=10,
            size=200,
            marks={
            10: {'label': '10', 'style': {'color': 'ffffff'} },
            20: {'label': '20', 'style': {'color': 'ffffff'} },
            30: {'label': '30', 'style': {'color': 'ffffff'} },
            40: {'label': '40', 'style': {'color': 'ffffff'} },
            50: {'label': '50', 'style': {'color': 'ffffff'} }
                }
            )
        ], style={'float': 'right', 'margin-right': '10%'}),
    #], style={ 'margin-left': '1%', 'margin-top': '1%', 'float':'left'}),
    html.Br(),
    html.Div(
        children=[html.Button(id='submit-button', n_clicks=0, children='Get recommendations')],
        style={ 'margin-left': '1%', 'margin-top': '4%', 'display': 'block', 'text-align': 'center'}),
    html.Div(id='output', style={ 'margin-left': '3%', 'margin-top': '3%,'})
 ])#, style={'padding': '50px'})

      
    #html.Div(
    #    id='dark-theme-component',
    #    children=[
    #        daq.DarkThemeProvider(theme=theme, children=
    ##                              daq.Knob(value=6))
     #   ],
    #    style={'display': 'block', 'margin-left': 'calc(50% - 110px)'}
    #)

#@app.callback(
#    Output('main', 'children'),
#    [Input('daq-light-dark-theme', 'value')]
#)
def turn_dark(dark_theme): 
    if(dark_theme):
        theme.update(
            dark=True
        )
    else:
        theme.update(
            dark=False
        )
    return daq.DarkThemeProvider(theme=theme, children=
                   html.Div(id='output')              )

#@app.callback(
#    Output('main', 'style'),
#    [Input('daq-light-dark-theme', 'value')]
#)
def change_bg(dark_theme):
    if(dark_theme):
        return {'background-color': '#303030', 'color': 'white'}
    else:
        return {'background-color': 'white', 'color': 'black'}

@app.callback(
    Output('output', 'children'),
    [Input('submit-button', 'n_clicks'),
     Input('numrecommendations', 'value'),
     Input('upperlimit', 'value'),
     Input('lowerlimit', 'value')],
    [State('context-state', 'value')])
def recommend(n_clicks, num_recs, upperlimit, lowerlimit, input_box):
    """ Wrapped function which takes user input in a text box, a slider and 2 on-off switches
    and returns a set of citation recommendations based on these parameters.
    ARGUMENTS: n_clicks: a parameter of the HTML button which indicates it has 
               been clicked
               input_box: the content of the text area in which the user has 
               entered a citation context.
               num_recs: no. of recommendations to return
               lowerlimit: if selected, the MAG50 model is used. MAG model is used by default
               upperlimit: if selected, recommendations with >500 citations are discarded.

    RETURNS:   list of recommendations with titles displayed, abstract, no. of citations and year
               in tooltip. Each recommendation links to the corresponding MAG page"""

    context = clean_text(input_box)
    print(upperlimit, num_recs, n_clicks)
    if context != '':
        if lowerlimit:
            hd2vrecommendations = hd2v_wvindvout_recommend(context, hd2vreducedmodel)        
            bm25recommendations = solr_recommend(context, 'mag_en_cs_50_all')
            citedbm25_recommendations = solr_cited_recommend(context, 'mag_en_cs_50_cited_all')
            if not hd2vrecommendations or not bm25recommendations or not citedbm25_recommendations:
                return html.Div([
                    html.Br(),
                    html.Br(),
                    html.H2('No recommendations returned.'),
                ])
            hybrid_recommendations = hybrid_recommend(hd2vrecommendations, bm25recommendations, citedbm25_recommendations)
            # magid, title, year, citations, abstract
            if upperlimit:
                all_recommendations = get_paper_details(hybrid_recommendations)
                reduced_recommendations = [recomm for recomm in all_recommendations if recomm[3]<=500]
                reduced_recommendations = get_topn(reduced_recommendations, num_recs)
            else:
                reduced_recommendations = get_paper_details(get_topn(hybrid_recommendations, num_recs))
            #recommended_titles = [details[1] for details in get_paper_details(reduced_recommendations)]
            return html.Div([
                    html.Br(),
                    html.Br(),
                    html.H2('Recommendations:'),
                    html.Ol([html.Li(html.A(recomm[1], 
                                            href='https://academic.microsoft.com/paper/{}'.format(recomm[0]),
                                            title=' Year: {}\nAbstract:{}'\
                                                .format(recomm[2], recomm[4]))
                                    ) 
                        for recomm in reduced_recommendations])
                ])
        else:
            hd2vrecommendations = hd2v_wvindvout_recommend(context, hd2vmodel)
            bm25recommendations = solr_recommend(context, 'mag_en_cs_all')
            citedbm25_recommendations = solr_cited_recommend(context, 'mag_en_cs_cited_all')
            if not hd2vrecommendations or not bm25recommendations or not citedbm25_recommendations:
                return html.Div([
                    html.Br(),
                    html.Br(),
                    html.H2('No recommendations returned.'),
                ])
            hybrid_recommendations = hybrid_recommend(hd2vrecommendations, bm25recommendations, citedbm25_recommendations)
            # magid, title, year, citations, abstract
            if upperlimit:
                all_recommendations = get_paper_details(hybrid_recommendations)
                reduced_recommendations = [recomm for recomm in all_recommendations if recomm[3]<=500]
                reduced_recommendations = get_topn(reduced_recommendations, num_recs)
            else:
                #print(hybrid_recommendations)
                reduced_recommendations = get_paper_details(get_topn(hybrid_recommendations, num_recs))
            #recommended_titles = [details[1] for details in get_paper_details(reduced_recommendations)]
            return html.Div([
                    html.Br(),
                    html.Br(),
                    html.H2('Recommendations:'),
                    html.Ol([html.Li(html.A(recomm[1], 
                                            href='https://academic.microsoft.com/paper/{}'.format(recomm[0]),
                                            title=' Year: {}\nAbstract:{}'\
                                                .format(recomm[2], recomm[4]))
                                    ) 
                        for recomm in reduced_recommendations])
            ])


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
