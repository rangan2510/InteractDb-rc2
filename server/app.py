import time, requests, urllib3, os, traceback, json
import pandas as pd
from guid import GUID
from flask import Flask, request, session, render_template, redirect, url_for, jsonify, send_from_directory
from flask_cors import CORS
import urllib
from itertools import product, combinations
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from pyvis.network import Network
import sqlalchemy as sa
import graphistry
import graph_visualization as gv

params = urllib.parse.quote_plus('Driver={ODBC Driver 17 for SQL Server};'
                      'Server=localhost;'
                      'Database=InteractDb_rc2;'
                      'Trusted_Connection=yes;')

engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
cnxn = engine.connect()

# %%
net = Network(directed=True, notebook=True, cdn_resources='remote')
net.set_edge_smooth('dynamic')
options = '''
const options = {
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -16150,
      "centralGravity": 0,
      "springLength": 155
    },
    "minVelocity": 0.75
  }
}
'''
net.set_options(options)

def get_drugbank_ids(syn):
    qry = 'SELECT * from drug_nodes WHERE synonyms LIKE \'%' +  syn + '%\''
    df = pd.read_sql_query(qry, cnxn)
    ids = None
    if len(df) > 0:
        ids = df['drugbank_id'].to_list()
    return ids


app = Flask(__name__)
app.secret_key = GUID()
app.config.from_object("config.Config")
CORS(app)


@app.route("/api/search")
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

@app.route("/api/interactions")
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

@app.route("/api/drug-pair-vis")
def db_ens_uni_graph():
  x = request.args.get('x')
  y = request.args.get('y')
  dnames = [x,y]
  gene_df, gene_search_space_, gene_df_list, dfs = gv.run_queries(dnames, cnxn)
  nodes_df = gv.create_nodes_list(dnames, gene_df, gene_search_space_)
  edges_df = gv.create_edge_list(dnames, gene_df, gene_df_list, dfs)
  url, graph_obj = gv.graph_vis(edges_df, nodes_df)
  return jsonify(url)

@app.route("/api/drug-set-se")
def side_effects():
    q = request.args.get('q')
    x = q.split("|")

    x = [i.strip() for i in x]
    
    sql_x = []
    for items in x:
        x_ = '%' + items + '%'
        sql_x.append(x_)

    sql_x = ' OR '.join(sql_x)

    print(sql_x)

    qry = '''
    SELECT Drug.name AS [Drug Name], Drug.synonyms, Drug.atc AS [atc], SideEffect.name AS [SideEffect]
    FROM drug_nodes Drug, drug_sideeffect_edges DSE, sideeffect_nodes SideEffect
    WHERE MATCH(Drug-(DSE)->SideEffect)
    AND CONTAINS(synonyms, '{0}')
    '''.format(sql_x)
    
    df_x = pd.read_sql_query(qry, cnxn)

    payload = []

    for idx, row in df_x.iterrows():
        payload.append(row.to_dict())
    
    return jsonify(payload)



@app.route("/")
def index():
    return jsonify("InteractDb_rc2 build_2310.01")


if __name__ == '__main__':
    app.run()
# %%
