import time, requests, urllib3, os, traceback
import pandas as pd
from guid import GUID
from flask import Flask, request, session, render_template, redirect, url_for, jsonify
from flask_cors import CORS
import urllib
from itertools import product
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import sqlalchemy as sa

params = urllib.parse.quote_plus('Driver={ODBC Driver 17 for SQL Server};'
                      'Server=localhost;'
                      'Database=InteractDb_rc2;'
                      'Trusted_Connection=yes;')

engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
cnxn = engine.connect()


app = Flask(__name__)
app.secret_key = GUID()
app.config.from_object("config.Config")
CORS(app)


@app.route("/search")
def basic_synonym_search():
    q = request.args.get('q')
    mode=request.args.get('mode')

    if mode=="atc":
        qry = 'SELECT * from drug_nodes WHERE atc LIKE \'%' + str(q).upper() + '%\''
        df = pd.read_sql_query(qry, cnxn)
    else:
        qry = 'SELECT * from drug_nodes WHERE synonyms LIKE \'%' +  q + '%\''
        df = pd.read_sql_query(qry, cnxn)
        
    collist = df.columns.tolist()
    # you can now select from this list any arbritrary range
    df = df[collist[1:]] 
    return df.to_json(), 200

@app.route("/interactions")
def easy_interactions_search():
    x = request.args.get('x')
    y = request.args.get('y')
    mode=request.args.get('mode')

    if mode=="atc":
        qry = 'SELECT * from drug_nodes WHERE atc LIKE \'%' + str(x).upper() + '%\''
        df_x = pd.read_sql_query(qry, cnxn)
        x_ids = df_x['drugbank_id'].to_list()

        qry = 'SELECT * from drug_nodes WHERE atc LIKE \'%' +  str(y).upper() + '%\''
        df_y = pd.read_sql_query(qry, cnxn)
        y_ids = df_y['drugbank_id'].to_list()
    else:
        #get x list:
        qry = 'SELECT * from drug_nodes WHERE synonyms LIKE \'%' +  str(x) + '%\''
        df_x = pd.read_sql_query(qry, cnxn)
        x_ids = df_x['drugbank_id'].to_list()
        
        #get y list
        qry = 'SELECT * from drug_nodes WHERE synonyms LIKE \'%' +  str(y) + '%\''
        df_y = pd.read_sql_query(qry, cnxn)
        y_ids = df_y['drugbank_id'].to_list()

    search_space = list(product(x_ids, y_ids))

    int_df_list = []
    for i,j in search_space:
        qry = "SELECT * from drug_drug_edges WHERE src = \'" + i + "\' AND dst = \'" + j + "\'"
        df = pd.read_sql_query(qry, cnxn)
        if len(df)>0:
            collist = df.columns.tolist()
            df = df[collist[3:]] 
            int_df_list.append(df)

    if len(int_df_list)>0:
        res = pd.concat(int_df_list)
        res.reset_index(drop=True, inplace=True)
        return res.to_json(), 200
    else:
        return jsonify("No interactions found."), 200


@app.route("/")
def index():
    return jsonify("InteractDb_rc2 build_230305.01")


if __name__ == '__main__':
    app.run()