USE [InteractDb_rc2]
GO

DROP TABLE IF EXISTS [dbo].[disease_nodes]

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[disease_nodes](
	[disgenet_id] [nvarchar](50) NOT NULL,
	[name] [nvarchar](max) NOT NULL,
	[vocabulary] [nvarchar](50) NOT NULL,
	[vocab_code] [nvarchar](50) NOT NULL,
	[vocab_name] [nvarchar](max) NOT NULL,
	[disease_type] [nvarchar](50) NOT NULL,
	[MeSH_tree_number] [nvarchar](50) NOT NULL,
	[MeSH_heading] [nvarchar](max) NOT NULL
) AS NODE
GO

INSERT INTO disease_nodes
SELECT * FROM [InteractDb_rc2].[dbo].[temp]

SELECT * FROM disease_nodes