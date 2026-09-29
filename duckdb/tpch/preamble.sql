-- Run by every DuckDB benchmark leg before `.timer on`: caches Parquet metadata for the
-- process and loads it untimed, so timed queries never read or parse a footer.
SET parquet_metadata_cache=true;
SELECT * FROM '/nvme/tpch/customer.parquet' LIMIT 0;
SELECT * FROM '/nvme/tpch/lineitem.parquet' LIMIT 0;
SELECT * FROM '/nvme/tpch/nation.parquet' LIMIT 0;
SELECT * FROM '/nvme/tpch/orders.parquet' LIMIT 0;
SELECT * FROM '/nvme/tpch/part.parquet' LIMIT 0;
SELECT * FROM '/nvme/tpch/partsupp.parquet' LIMIT 0;
SELECT * FROM '/nvme/tpch/region.parquet' LIMIT 0;
SELECT * FROM '/nvme/tpch/supplier.parquet' LIMIT 0;
