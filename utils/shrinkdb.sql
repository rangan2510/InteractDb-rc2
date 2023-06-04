USE InteractDb_rc2
select name,recovery_model_desc from sys.databases
ALTER DATABASE InteractDb_rc2 SET RECOVERY simple
DBCC SHRINKFILE (InteractDb_rc2_log , 1)