SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
DROP TABLE IF EXISTS [dbo].[disease_gene_edges]
GO


CREATE TABLE [dbo].[disease_gene_edges](
	[src] [nvarchar](50) NOT NULL,
	[dst] [nvarchar](50) NOT NULL,
	[gene_name] [nvarchar](50) NULL,
	[gene_description] [nvarchar](max) NULL,
	[pLI] [float] NULL,
	[DSI] [float] NULL,
	[DPI] [float] NULL,
	[source] [nvarchar](50) NULL,
	[association_type] [nvarchar](50) NULL,
	[sentence] [nvarchar](max) NULL,
	[pmid] [float] NULL
) AS EDGE
GO

INSERT INTO disease_gene_edges
SELECT TRIM('"' FROM REPLACE([from_id], '""','"')) AS [from_id]
      ,TRIM('"' FROM REPLACE([to_id], '""','"')) AS [to_id]
      ,[src]
      ,[dst]
      ,[gene_name]
      ,[gene_description]
      ,[pLI]
      ,[DSI]
      ,[DPI]
      ,[source]
      ,[association_type]
      ,[sentence]
      ,[pmid]
  FROM [InteractDb_rc2].[dbo].[temp]

SELECT * FROM disease_gene_edges