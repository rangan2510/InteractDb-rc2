SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
DROP TABLE IF EXISTS [dbo].[drug_disease_edges]
GO


CREATE TABLE [dbo].[drug_disease_edges](
	[src] [nvarchar](50) NOT NULL,
	[dst] [nvarchar](50) NOT NULL,
	[disgenet_name] [nvarchar](max) NOT NULL,
	[assoc_cond] [nvarchar](max) NOT NULL,
	[icd11Code] [nvarchar](50) NOT NULL,
	[icd11Title] [nvarchar](max) NOT NULL,
	[icd10Code] [nvarchar](50) NOT NULL,
	[icd10Title] [nvarchar](max) NOT NULL
) AS EDGE
GO


INSERT INTO [dbo].[drug_disease_edges]
SELECT TRIM('"' FROM REPLACE([from_id], '""','"')) AS [from_id]
      ,TRIM('"' FROM REPLACE([to_id], '""','"')) AS [to_id]
      ,[src]
      ,[dst]
      ,[disgenet_name]
      ,[assoc_cond]
      ,[icd11Code]
      ,[icd11Title]
      ,[icd10Code]
      ,[icd10Title]
  FROM [InteractDb_rc2].[dbo].[temp]

  SELECT * FROM [dbo].[drug_disease_edges]