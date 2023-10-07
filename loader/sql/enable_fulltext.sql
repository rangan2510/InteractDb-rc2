USE InteractDb_rc2

CREATE FULLTEXT CATALOG ftCatalog AS DEFAULT;  
GO  

CREATE UNIQUE INDEX ui_drugbanksyn ON drug_nodes(drugbank_id)

CREATE FULLTEXT INDEX ON drug_nodes(synonyms)
   KEY INDEX ui_drugbanksyn
   WITH STOPLIST = SYSTEM;
GO