USE [InteractDb_rc2]
GO
/****** Object:  Table [dbo].[gene_gene_edges]    Script Date: 5/31/2023 7:36:01 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

DROP TABLE IF EXISTS gene_gene_edges

CREATE TABLE [dbo].[gene_gene_edges](
	[src] [nvarchar](50) NOT NULL,
	[dst] [nvarchar](50) NOT NULL,
	[neighborhood] [smallint] NOT NULL,
	[fusion] [smallint] NOT NULL,
	[cooccurence] [smallint] NOT NULL,
	[coexpression] [smallint] NOT NULL,
	[experimental] [smallint] NOT NULL,
	[database] [smallint] NOT NULL,
	[textmining] [smallint] NOT NULL,
	[combined_score] [smallint] NOT NULL
) AS EDGE
GO

INSERT INTO gene_gene_edges
SELECT TRIM('"' FROM REPLACE([from_id], '""','"')) AS [from_id]
      ,TRIM('"' FROM REPLACE([to_id], '""','"')) AS [to_id]
      ,[src]
      ,[dst]
      ,[neighborhood]
      ,[fusion]
      ,[cooccurence]
      ,[coexpression]
      ,[experimental]
      ,[database]
      ,[textmining]
      ,[combined_score]
  FROM [InteractDb_rc2].[dbo].[temp]

SELECT * FROM gene_gene_edges