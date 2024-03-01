from Bio import SeqIO
import pandas as pd
import re

dir_path = "/home/swadesh/Desktop/Projects/UM-RD/France/InteractDb-rc2/data/drugs/drugbank_5.1.10/"
rows = []
with open(dir_path + "target_sequences_protein.fasta") as fasta_file:
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        target_id = re.findall("[\w]*$", seq_record.id)[0]
        for db_id in re.findall("DB[\d]*", seq_record.description):
            rows.append([target_id, db_id])

df = pd.DataFrame(rows, columns=["UID", "DBID"])
