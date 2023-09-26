import pandas as pd
from itertools import combinations
import sqlalchemy as sa
import pyodbc
import graphistry

def run_queries(dnames, con):
  gene_df_list = []
  dbid_df_list = []

  for dname in dnames:
    qry2 = 'SELECT TOP 1 drugbank_id from drug_nodes WHERE synonyms LIKE \'%' +  dname + '%\' '
    df2 = pd.read_sql(sa.text(qry2), con)
    dbid_df_list.append(df2)
  
  dbid_df = pd.concat(dbid_df_list)
  dbids = dbid_df['drugbank_id'].tolist()

  for dbid in dbids:
    qry = '''SELECT DISTINCT ensembl_id AS 'ensembl_id', gene_names AS 'gene_names' FROM gene_nodes WHERE uniprotkb_id IN (SELECT uniprotkb_id FROM drug_gene_edges WHERE drugbank_id = '{x}')'''.format(x=dbid)
    df = pd.read_sql(sa.text(qry), con)
    gene_df_list.append(df)

  gene_df = pd.concat(gene_df_list)
  gene_df = gene_df.drop_duplicates().reset_index(drop=True)

  gene_search_space_ = list(set(gene_df["ensembl_id"].unique()))
  gene_search_space = list(combinations(gene_search_space_, 2))

  dfs = []

  for g in gene_search_space:
          x_, y_ = g
          qry = '''\
          SELECT Gene1.ensembl_id AS 'Gene 1', Gene2.ensembl_id AS 'Gene 2'
          FROM gene_nodes Gene1, gene_gene_edges, gene_nodes Gene2
          WHERE MATCH(Gene1-(gene_gene_edges)->Gene2)
          AND Gene1.ensembl_id = '{x}'
          AND Gene2.ensembl_id = '{y}'\
          '''.format(x=x_, y=y_)
          df = pd.read_sql(sa.text(qry), con)
          dfs.append(df)

  dfs = pd.concat(dfs)
  dfs = dfs.reset_index(drop=True)

  return gene_df, gene_search_space_, gene_df_list, dfs

def convert_to_int64_hex(rgb_hex):
    int64_hex = '0x' + rgb_hex[1:]+'00'
    return int(int64_hex, 16)

def create_nodes_list(dnames, gene_df, gene_search_space_):
  nodes_list = []
  
  for drug in dnames:
      nodes_list.append({'label':drug, 'color':"#DA03B3", 'size':20})

  for gene in gene_df['gene_names'].tolist():
      nodes_list.append({'label':gene, 'color':"#008080", 'size':10})

  for gene_grp in gene_search_space_:
      nodes_list.append({'label':gene_grp, 'color':"#FFA500", 'size':20})

  n_df = pd.DataFrame(nodes_list)
  n_df['int64_color'] = n_df['color'].apply(convert_to_int64_hex)
  return n_df

def create_edge_list(dnames, gene_df, gene_df_list, dfs):
  edge_list = []
  for i in range(len(gene_df_list)):
      for genes in gene_df_list[i]['ensembl_id'].unique():
          x = dnames[i]
          y = genes
          edge_list.append({'src':x, 'dst':y, 'color':"#DA03B3"})

  for idx, row in gene_df.iterrows():
      x, y = row["gene_names"], row["ensembl_id"]
      edge_list.append({'src':x, 'dst':y, 'color':"#008080"})

  for idx, row in dfs.iterrows():
      x, y = row["Gene 1"], row["Gene 2"]
      edge_list.append({'src':x, 'dst':y, 'color':"#FFA500"})

  e_df = pd.DataFrame(edge_list)
  e_df['int64_color'] = e_df['color'].apply(convert_to_int64_hex)
  return e_df

def graph_vis(e_df, n_df):
  graphistry.register(api=3, protocol="https", server="hub.graphistry.com", personal_key_id="BXPKW5G4ZE", personal_key_secret="IWR1SAP5XZGCAOFZ") 

  gx = graphistry.edges(e_df, 'src', 'dst').bind(edge_color='int64_color').nodes(n_df, 'label').bind(point_color='int64_color')
  gx = gx.settings(url_params={'pointSize': 0.5}).encode_point_size('size')

  public_url1 = gx.plot(render=False)
  return public_url1, gx
