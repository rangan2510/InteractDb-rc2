import time, requests, urllib3, os, traceback
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

@app.route("/graphvis")
def db_ens_uni_graph():
  x = request.args.get('x')
  y = request.args.get('y')
  dbids = [x,y]
  dname_df, gene_df, gene_search_space_, gene_df_list, dfs = gv.run_queries(dbids, cnxn)
  nodes_df = gv.create_nodes_list(dname_df, gene_df, gene_search_space_)
  edges_df = gv.create_edge_list(dname_df, gene_df, gene_df_list, dfs)
  url, graph_obj = gv.graph_vis(edges_df, nodes_df)
  return url

@app.route("/vis")
def visualize_ddi():
    x = request.args.get('x')
    y = request.args.get('y')

    # %%
    x_ids, y_ids = get_drugbank_ids(x), get_drugbank_ids(y)
    search_set = x_ids + y_ids
    search_set = list(set(search_set))
    search_space = list(product(x_ids, y_ids))


    # %%
    drug_group_id = 1
    gene_group_id = 2
    disease_group_id = 3

    # %% [markdown]
    # # Direct drug interactions

    # %%
    dfs = []

    for s in search_space:
        x_, y_ = s
        qry = '''\
        SELECT Drug1.drugbank_id AS src_id, Drug1.name AS 'Source Drug', Drug2.drugbank_id as dst_id, Drug2.name AS 'Interacting Drug', [dbo].[drug_drug_edges].[desc] AS 'Effect of Interaction'
        FROM drug_nodes Drug1, drug_drug_edges, drug_nodes Drug2
        WHERE MATCH(Drug1-(drug_drug_edges)->Drug2)
        AND Drug1.drugbank_id = '{x}'
        AND Drug2.drugbank_id = '{y}'\
        '''.format(x=x_, y=y_)
        df = pd.read_sql_query(qry, cnxn)
        dfs.append(df)

    dfs = pd.concat(dfs)



    # %%
    for idx, row in dfs.iterrows():
        x, y = row["Source Drug"], row["Interacting Drug"]
        x_id, y_id = row["src_id"], row["dst_id"]
        net.add_node(x_id, title=x_id, label=x, group=drug_group_id)
        net.add_node(y_id, title=y_id, label=y, group=drug_group_id)
        net.add_edge(x_id,y_id, title=str(row["Effect of Interaction"]))



    # %%
    dfs = []
    for item in search_set:
        qry = '''\
        SELECT Drug1.drugbank_id, Drug1.name AS [Drug 1], Genes.gene_names, Genes.ensembl_id, drug_gene_edges.action
        FROM drug_nodes Drug1, drug_gene_edges, gene_nodes Genes
        WHERE MATCH(Drug1-(drug_gene_edges)->Genes)
        AND Drug1.drugbank_id = '{x}'\
        '''.format(x=item)
        df = pd.read_sql_query(qry, cnxn)
        dfs.append(df)

    dfs = pd.concat(dfs)


    # %%
    # dfg = pd.DataFrame(dfs.groupby('ensembl_id')['gene_names'].apply(', '.join))
    # dfg = pd.DataFrame(dfg.to_records())
    # dfs = pd.merge(left=dfg, right=dfs[["drugbank_id", "Drug 1", "ensembl_id", "action"]], on="ensembl_id", how="inner")
    # dfs = dfs.drop_duplicates()

    # %%
    for idx, row in dfs.iterrows():
        x, y, z = row["Drug 1"], row["gene_names"], row["action"]
        x_id, y_id = row["drugbank_id"], row["gene_names"]
        net.add_node(x_id, title=x_id, label=x, group=drug_group_id)
        net.add_node(y_id, title=y_id, label=y, group=gene_group_id)
        net.add_edge(x_id,y_id, title=str(z))

    # %% [markdown]
    # # Gene Gene interactions

    # %%
    gene_search_space_ = list(set(dfs["ensembl_id"].to_list()))
    gene_search_space = list(combinations(gene_search_space_, 2))


    # %%
    dfs = []

    for g in gene_search_space:
        x_, y_ = g
        qry = '''\
        SELECT Gene1.gene_names AS 'Gene 1', Gene2.gene_names AS 'Gene 2'
        FROM gene_nodes Gene1, gene_gene_edges, gene_nodes Gene2
        WHERE MATCH(Gene1-(gene_gene_edges)->Gene2)
        AND Gene1.ensembl_id = '{x}'
        AND Gene2.ensembl_id = '{y}'\
        '''.format(x=x_, y=y_)
        df = pd.read_sql_query(qry, cnxn)
        dfs.append(df)

    dfs = pd.concat(dfs)


    # %%
    for idx, row in dfs.iterrows():
        x, y = row["Gene 1"], row["Gene 2"]
        net.add_edge(x,y)


    # %%
    dfs = []

    for g in gene_search_space_:
        qry = '''\
        SELECT Disease.name AS [Disease Name], Disease.vocabulary, Genes.gene_names as [Associated Genes], disease_gene_edges.sentence
        FROM disease_nodes Disease, disease_gene_edges, gene_nodes Genes
        WHERE MATCH (Genes-(disease_gene_edges)->Disease)
        AND Genes.ensembl_id = '{x}'\
            '''.format(x=g)
        df = pd.read_sql_query(qry, cnxn)
        dfs.append(df)

    dfs = pd.concat(dfs)

    # %%
    dfs = dfs.drop_duplicates()


    # %%
    for idx, row in dfs.iterrows():
        dname = row["Disease Name"]
        gene = row["Associated Genes"]
        net.add_node(dname, title=dname, label=dname, group=disease_group_id)
        net.add_edge(dname, gene, title=str(row["sentence"]))

    # %% [markdown]
    # ## with drugs

    # %%
    dfs = []

    for d in search_set:
        qry = '''\
        SELECT Disease.vocab_name AS [Disease Name], Disease.MeSH_heading, Drugs.name as [Associated Drugs]
        FROM disease_nodes Disease, drug_disease_edges, drug_nodes Drugs
        WHERE MATCH (Drugs-(drug_disease_edges)->Disease)
        AND Drugs.drugbank_id = '{x}'\
            '''.format(x=d)
        df = pd.read_sql_query(qry, cnxn)
        dfs.append(df)

    dfs = pd.concat(dfs)


    # %%
    for e in net.edges:
        e['arrows'] = {'to': {'scaleFactor': 0.5}}
    #net.show_buttons(filter_=['physics'])
    net.show("demo.html")
    return(send_from_directory("../", "demo.html", as_attachment=False))



@app.route("/")
def index():
    return jsonify("InteractDb_rc2 build_2308.03")


if __name__ == '__main__':
    app.run()
# %%
