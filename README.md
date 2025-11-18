# etl-helper

**Advanced ETL tool for CSV/JSON/Parquet** with filtering, transformations, SQL querying, profiling, and validation —  
built in **Rust** using **Polars**, **Clap**, and **Indicatif**.

---

## 🚀 Features

### 🔧 Transform CSV Data
- Filter rows (`=`, `!=`, `<`, `>`, `>=`, `<=`, `~contains~`)
- Select columns by name
- Sample a fraction of the dataset (randomized)
- Drop rows with null values
- Auto-detect delimiter
- Optional **dry run** mode (preview output without writing)

---

### 🔄 Convert Between Formats
Convert seamlessly between:
- **CSV**
- **JSON / NDJSON / JSONL**
- **Parquet**

---

### 🧠 SQL Querying
Run SQL queries against CSV or Parquet files using Polars’ lazy query engine:

```sql
SELECT name, age FROM self WHERE age > 30
