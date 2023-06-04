SELECT Drug1.name, Drug2.name AS FriendName
FROM drug_nodes Drug1, drug_drug_edges, drug_nodes Drug2
WHERE MATCH(Drug1-(drug_drug_edges)->Drug2)
AND Drug1.drugbank_id = 'DB00001';