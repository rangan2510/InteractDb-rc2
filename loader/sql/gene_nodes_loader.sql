USE [InteractDb_rc2]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[gene_nodes](
	[ensembl_id] [nvarchar](50) NOT NULL,
	[uniprotkb_id] [nvarchar](50) NOT NULL,
	[gene_names] [nvarchar](50) NOT NULL,
	[is_known_drug_tgt] [bit] NOT NULL,
	[is_disease_associated] [bit] NOT NULL
) AS NODE
GO

INSERT INTO gene_nodes
SELECT [ensembl_id]
      ,[uniprotkb_id]
      ,[gene_names]
      ,[is_known_drug_tgt]
      ,[is_disease_associated]
  FROM [InteractDb_rc2].[dbo].[temp]

SELECT * FROM gene_nodes