SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
DROP TABLE IF EXISTS [dbo].[drug_sideeffect_edges]
GO
CREATE TABLE [dbo].[drug_sideeffect_edges](
	[atc] [nvarchar](50) NOT NULL,
	[stitch_id] [nvarchar](50) NOT NULL,
	[umls_cid] [nvarchar](50) NOT NULL,
	[freq_ub] [float] NULL,
	[freq_lb] [float] NULL
) AS EDGE
GO

INSERT INTO [dbo].[drug_sideeffect_edges]
SELECT TRIM('"' FROM REPLACE([from_id], '""','"')) AS [from_id]
      ,TRIM('"' FROM REPLACE([to_id], '""','"')) AS [to_id]
      ,[atc]
      ,[stitch_id]
      ,[umls_cid]
      ,[freq_ub]
      ,[freq_lb]
  FROM [InteractDb_rc2].[dbo].[temp]

SELECT * FROM [dbo].[drug_sideeffect_edges]