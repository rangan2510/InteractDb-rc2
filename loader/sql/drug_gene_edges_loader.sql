USE [InteractDb_rc2]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

DROP TABLE IF EXISTS drug_gene_edges
GO

CREATE TABLE [dbo].[drug_gene_edges](
	[drugbank_id] [nvarchar](50) NOT NULL,
	[uniprotkb_id] [nvarchar](50) NOT NULL,
	[action] [nvarchar](50) NOT NULL
) AS EDGE
GO

INSERT INTO drug_gene_edges
SELECT TRIM('"' FROM REPLACE([from_id], '""','"')) AS [from_id]
	  ,TRIM('"' FROM REPLACE([to_id], '""','"')) AS [to_id]
      ,[drugbank_id]
      ,[uniprotkb_id]
      ,[action]
  FROM [InteractDb_rc2].[dbo].[temp]

SELECT * FROM drug_gene_edges