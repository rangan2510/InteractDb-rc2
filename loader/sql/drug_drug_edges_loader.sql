USE [InteractDb_rc2]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

DROP TABLE IF EXISTS drug_drug_edges

CREATE TABLE drug_drug_edges(
	[src] [nvarchar](50) NOT NULL,
	[dst] [nvarchar](50) NOT NULL,
	[desc] [varchar](max) NOT NULL
) AS EDGE
GO


SELECT TRIM('"' FROM REPLACE([from_id], '""','"')) AS [from_id]
      ,TRIM('"' FROM REPLACE([to_id], '""','"')) AS [to_id]
      ,[src]
      ,[dst]
      ,[desc]
  FROM [InteractDb_rc2].[dbo].[temp]

INSERT INTO drug_drug_edges
	SELECT TRIM('"' FROM REPLACE([from_id], '""','"')) AS [from_id]
		  ,TRIM('"' FROM REPLACE([to_id], '""','"')) AS [to_id]
		  ,[src]
		  ,[dst]
		  ,[desc]
	  FROM [InteractDb_rc2].[dbo].[temp]


SELECT * FROM drug_drug_edges

