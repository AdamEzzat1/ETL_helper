use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// CLI DEFINITION
// ============================================================================

/// Advanced ETL tool for CSV/JSON/Parquet with filtering, transformations, and conversions
#[derive(Parser, Debug)]
#[command(name = "etl-helper")]
#[command(version, about, long_about = None)]
#[command(after_help = "EXAMPLES:\n  \
    # Transform CSV with filtering\n  \
    etl-helper transform -i data.csv -o out.csv --filter 'status=active' --filter 'age>25'\n\n  \
    # Convert formats\n  \
    etl-helper convert -i data.json -o data.csv --from json --to csv\n\n  \
    # Query with SQL\n  \
    etl-helper query -i data.csv -o result.csv --sql 'SELECT name, age FROM self WHERE age > 30'\n\n  \
    # Profile data\n  \
    etl-helper profile -i data.csv -o report.json\n\n  \
    # Sample data\n  \
    etl-helper transform -i large.csv -o sample.csv --sample 0.1")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output for debugging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Transform a CSV by filtering, selecting columns, and sampling
    Transform {
        /// Input CSV file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output CSV file path
        #[arg(short, long)]
        output: PathBuf,

        /// Comma-separated list of columns to keep (e.g. "name,age,salary")
        #[arg(short = 'c', long)]
        columns: Option<String>,

        /// Filters: column=value, column>value, column<value, column>=value, column<=value,
        /// column!=value, column~contains~value
        /// Example: --filter "country=US" --filter "age>25" --filter "name~contains~Smith"
        #[arg(short, long)]
        filter: Vec<String>,

        /// Drop rows where any selected columns have null values
        #[arg(long, default_value_t = false)]
        drop_empty: bool,

        /// CSV delimiter (default: comma). Example: --delimiter ';'
        #[arg(long, default_value_t = ',')]
        delimiter: char,

        /// Auto-detect CSV delimiter from first line (overrides --delimiter if set)
        #[arg(long, default_value_t = false)]
        auto_delimiter: bool,

        /// Sample fraction (0.0 to 1.0). Example: --sample 0.1 for 10%
        #[arg(long)]
        sample: Option<f64>,

        /// Show statistics summary after transformation
        #[arg(long, default_value_t = true)]
        stats: bool,

        /// Dry run - show what would be done without writing output
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },

    /// Convert between formats (CSV, JSON, Parquet)
    Convert {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Input format
        #[arg(long)]
        from: Format,

        /// Output format
        #[arg(long)]
        to: Format,

        /// CSV delimiter (only for CSV input/output)
        #[arg(long)]
        delimiter: Option<char>,

        /// Auto-detect CSV delimiter
        #[arg(long, default_value_t = false)]
        auto_delimiter: bool,
    },

    /// Query data using SQL
    Query {
        /// Input file path (CSV or Parquet)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// SQL query (use 'self' as table name)
        #[arg(short, long)]
        sql: String,

        /// CSV delimiter (only for CSV input/output)
        #[arg(long)]
        delimiter: Option<char>,

        /// Auto-detect CSV delimiter
        #[arg(long, default_value_t = false)]
        auto_delimiter: bool,
    },

    /// Profile data and generate statistics report
    Profile {
        /// Input file path (CSV or Parquet)
        #[arg(short, long)]
        input: PathBuf,

        /// Output JSON report file (if omitted, prints to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// CSV delimiter (only for CSV input)
        #[arg(long)]
        delimiter: Option<char>,

        /// Auto-detect CSV delimiter
        #[arg(long, default_value_t = false)]
        auto_delimiter: bool,
    },

    /// Validate data against optional schema rules
    Validate {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Optional JSON schema file
        #[arg(short, long)]
        schema: Option<PathBuf>,

        /// Output validation report
        #[arg(short = 'r', long)]
        report: PathBuf,

        /// CSV delimiter (only for CSV input)
        #[arg(long)]
        delimiter: Option<char>,

        /// Auto-detect CSV delimiter
        #[arg(long, default_value_t = false)]
        auto_delimiter: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Format {
    Csv,
    Json,
    Parquet,
}

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(thiserror::Error, Debug)]
pub enum EtlError {
    #[error("Column '{0}' not found in headers. Available columns: {1}")]
    ColumnNotFound(String, String),

    #[error("Invalid filter format: '{0}'. Expected format: column=value, column>value, column~contains~value")]
    InvalidFilter(String),

    #[error("Invalid sample rate: {0}. Must be between 0.0 and 1.0")]
    InvalidSampleRate(f64),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

// ============================================================================
// FILTER PARSING
// ============================================================================

#[derive(Debug, Clone)]
enum FilterOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    Contains,
}

#[derive(Debug, Clone)]
struct FilterCond {
    column: String,
    op: FilterOp,
    value: String,
}

fn parse_filters(raw_filters: &[String]) -> Result<Vec<FilterCond>> {
    let mut filters = Vec::new();

    for f in raw_filters {
        // Check for ~contains~
        if let Some(pos) = f.find("~contains~") {
            let column = f[..pos].trim().to_string();
            let value = f[pos + 10..].trim().to_string();
            filters.push(FilterCond {
                column,
                op: FilterOp::Contains,
                value,
            });
            continue;
        }

        // Check for other operators
        let (op, delim) = if f.contains(">=") {
            (FilterOp::Gte, ">=")
        } else if f.contains("<=") {
            (FilterOp::Lte, "<=")
        } else if f.contains("!=") {
            (FilterOp::Ne, "!=")
        } else if f.contains('>') {
            (FilterOp::Gt, ">")
        } else if f.contains('<') {
            (FilterOp::Lt, "<")
        } else if f.contains('=') {
            (FilterOp::Eq, "=")
        } else {
            return Err(EtlError::InvalidFilter(f.clone()).into());
        };

        let parts: Vec<&str> = f.splitn(2, delim).collect();
        if parts.len() != 2 {
            return Err(EtlError::InvalidFilter(f.clone()).into());
        }

        filters.push(FilterCond {
            column: parts[0].trim().to_string(),
            op,
            value: parts[1].trim().to_string(),
        });
    }

    Ok(filters)
}

// ============================================================================
// UTILITIES
// ============================================================================

fn detect_delimiter(path: &PathBuf) -> Result<char> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open file for delimiter detection: {:?}", path))?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    let bytes_read = reader
        .read_line(&mut first_line)
        .context("Failed to read first line for delimiter detection")?;

    if bytes_read == 0 {
        println!("⚠️  File appears empty; falling back to comma delimiter.");
        return Ok(',');
    }

    let candidates = [',', ';', '\t', '|'];
    let mut best = (',', 0usize);

    for &cand in &candidates {
        let count = first_line.chars().filter(|&c| c == cand).count();
        if count > best.1 {
            best = (cand, count);
        }
    }

    if best.1 == 0 {
        println!("⚠️  No obvious delimiter found; falling back to comma.");
        Ok(',')
    } else {
        println!("✅ Detected '{}' as delimiter ({} occurrences)", best.0, best.1);
        Ok(best.0)
    }
}

fn create_progress_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_message(msg.to_string());
    pb.set_style(
        ProgressStyle::with_template("{spinner} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"]),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

fn print_stats(total_rows: usize, output_rows: usize, elapsed: Duration, verbose: bool) {
    println!("\n📊 Summary:");
    println!("  Rows in:  {}", total_rows);
    println!("  Rows out: {}", output_rows);

    if total_rows > 0 {
        let filtered = total_rows.saturating_sub(output_rows);
        let pct = (filtered as f64 / total_rows as f64) * 100.0;
        println!("  Filtered (incl. sampling): {} ({:.1}%)", filtered, pct);
    }

    println!("  Time:     {:.2}s", elapsed.as_secs_f64());

    if verbose && elapsed.as_secs_f64() > 0.0 {
        println!("  Rate:     {:.0} rows/sec", output_rows as f64 / elapsed.as_secs_f64());
    }
}

// ============================================================================
// DATA LOADING
// ============================================================================

fn load_dataframe(
    path: &PathBuf,
    delimiter: Option<char>,
    auto_delimiter: bool,
) -> Result<DataFrame> {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "csv" => {
            let delim = if let Some(d) = delimiter {
                d
            } else if auto_delimiter {
                detect_delimiter(path)?
            } else {
                ','
            };

            let file = File::open(path)
                .with_context(|| format!("Failed to open CSV file: {:?}", path))?;

            CsvReader::new(file)
                .has_header(true)
                .with_delimiter(delim as u8)
                .finish()
                .map_err(|e| anyhow::anyhow!("Polars CSV read error: {}", e))
        }
        "parquet" => {
            let file = File::open(path)
                .with_context(|| format!("Failed to open Parquet file: {:?}", path))?;

            ParquetReader::new(file)
                .finish()
                .map_err(|e| anyhow::anyhow!("Polars Parquet read error: {}", e))
        }
        "json" | "ndjson" | "jsonl" => {
            let file = File::open(path)
                .with_context(|| format!("Failed to open JSON file: {:?}", path))?;

            JsonReader::new(file)
                .finish()
                .map_err(|e| anyhow::anyhow!("Polars JSON read error: {}", e))
        }
        _ => Err(anyhow::anyhow!(
            "Unsupported file format: {}. Supported: csv, parquet, json",
            extension
        )),
    }
}

fn save_dataframe(
    df: &mut DataFrame,
    path: &PathBuf,
    format: Option<Format>,
    delimiter: Option<char>,
) -> Result<()> {
    let fmt = if let Some(f) = format {
        f
    } else {
        // Infer from extension
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "csv" => Format::Csv,
            "parquet" => Format::Parquet,
            "json" | "ndjson" | "jsonl" => Format::Json,
            _ => Format::Csv, // Default
        }
    };

    let file = File::create(path)
        .with_context(|| format!("Failed to create output file: {:?}", path))?;

    match fmt {
        Format::Csv => {
            let delim = delimiter.unwrap_or(',');
            CsvWriter::new(file)
                .has_header(true)
                .with_delimiter(delim as u8)
                .finish(df)
                .map_err(|e| anyhow::anyhow!("Polars CSV write error: {}", e))?;
        }
        Format::Parquet => {
            ParquetWriter::new(file)
                .finish(df)
                .map_err(|e| anyhow::anyhow!("Polars Parquet write error: {}", e))?;
        }
        Format::Json => {
            JsonWriter::new(file)
                .finish(df)
                .map_err(|e| anyhow::anyhow!("Polars JSON write error: {}", e))?;
        }
    }

    Ok(())
}

// ============================================================================
// TRANSFORM COMMAND
// ============================================================================

fn apply_filters(mut lf: LazyFrame, filters: &[FilterCond], verbose: bool) -> Result<LazyFrame> {
    if filters.is_empty() {
        return Ok(lf);
    }

    // Validate that all filter columns exist in schema (as best as we can)
    let schema = lf
        .schema()
        .map_err(|e| anyhow::anyhow!("Failed to infer schema for filters: {}", e))?;
    let available_cols: Vec<String> = schema
        .iter_fields()
        .map(|f| f.name().to_string())
        .collect();

    for filter in filters {
        if !schema.get(&filter.column).is_some() {
            return Err(EtlError::ColumnNotFound(
                filter.column.clone(),
                available_cols.join(", "),
            )
            .into());
        }
    }

    for filter in filters {
        if verbose {
            println!("  Applying filter: {} {:?} {}", filter.column, filter.op, filter.value);
        }

        let condition = match filter.op {
            FilterOp::Eq => {
                if let Ok(num) = filter.value.parse::<f64>() {
                    col(&filter.column).eq(lit(num))
                } else {
                    col(&filter.column).eq(lit(&filter.value))
                }
            }
            FilterOp::Ne => {
                if let Ok(num) = filter.value.parse::<f64>() {
                    col(&filter.column).neq(lit(num))
                } else {
                    col(&filter.column).neq(lit(&filter.value))
                }
            }
            FilterOp::Gt => {
                if let Ok(num) = filter.value.parse::<f64>() {
                    col(&filter.column).gt(lit(num))
                } else {
                    col(&filter.column).gt(lit(&filter.value))
                }
            }
            FilterOp::Lt => {
                if let Ok(num) = filter.value.parse::<f64>() {
                    col(&filter.column).lt(lit(num))
                } else {
                    col(&filter.column).lt(lit(&filter.value))
                }
            }
            FilterOp::Gte => {
                if let Ok(num) = filter.value.parse::<f64>() {
                    col(&filter.column).gt_eq(lit(num))
                } else {
                    col(&filter.column).gt_eq(lit(&filter.value))
                }
            }
            FilterOp::Lte => {
                if let Ok(num) = filter.value.parse::<f64>() {
                    col(&filter.column).lt_eq(lit(num))
                } else {
                    col(&filter.column).lt_eq(lit(&filter.value))
                }
            }
            FilterOp::Contains => col(&filter.column)
                .str()
                .contains(lit(&filter.value), true),
        };

        lf = lf.filter(condition);
    }

    Ok(lf)
}

fn transform_csv(
    input: PathBuf,
    output: PathBuf,
    columns: Option<String>,
    filters: Vec<FilterCond>,
    drop_empty: bool,
    delimiter: char,
    auto_delimiter: bool,
    sample: Option<f64>,
    show_stats: bool,
    dry_run: bool,
    verbose: bool,
) -> Result<()> {
    let start = Instant::now();

    // Validate sample rate
    if let Some(s) = sample {
        if s <= 0.0 || s > 1.0 {
            return Err(EtlError::InvalidSampleRate(s).into());
        }
    }

    let pb = create_progress_spinner("Loading and transforming data...");

    // Determine delimiter
    let final_delim = if auto_delimiter {
        detect_delimiter(&input)?
    } else {
        delimiter
    };

    // Parse columns into a reusable list
    let selected_cols: Option<Vec<String>> = columns.as_ref().map(|cols| {
        cols.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    });

    // Load data as LazyFrame for optimized execution
    let mut lf = LazyCsvReader::new(&input)
        .has_header(true)
        .with_delimiter(final_delim as u8)
        .finish()?;

    // Apply filters
    lf = apply_filters(lf, &filters, verbose)?;

    // Select columns
    if let Some(ref col_list) = selected_cols {
        let col_refs: Vec<&str> = col_list.iter().map(|s| s.as_str()).collect();
        if verbose {
            println!("  Selecting columns: {:?}", col_refs);
        }
        lf = lf.select(&[cols_to_expr(col_refs)]);
    }

    // Drop empty (null) values
    if drop_empty {
        if verbose {
            println!("  Dropping rows with null values");
        }

        lf = if let Some(ref col_list) = selected_cols {
            let col_refs: Vec<&str> = col_list.iter().map(|s| s.as_str()).collect();
            lf.drop_nulls(Some(col_refs))
        } else {
            lf.drop_nulls(None)
        };
    }

    // Collect to DataFrame
    let mut df = lf
        .collect()
        .map_err(|e| anyhow::anyhow!("Polars execution error: {}", e))?;

    let original_count = df.height();

    // Apply sampling if requested (random sample)
    if let Some(frac) = sample {
        if verbose {
            println!("  Sampling {:.1}% of rows", frac * 100.0);
        }
        df = df
            .sample_frac(frac, false, true, None)
            .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;
    }

    let output_count = df.height();

    pb.finish_and_clear();

    if dry_run {
        println!("🔍 DRY RUN - No files written");
        println!("  Would transform: {:?} → {:?}", input, output);
        println!("  Output rows: {}", output_count);
        return Ok(());
    }

    // Save output (CSV)
    save_dataframe(&mut df, &output, Some(Format::Csv), Some(final_delim))?;

    let elapsed = start.elapsed();

    println!("✅ Transform complete. Output: {:?}", output);

    if show_stats {
        print_stats(original_count, output_count, elapsed, verbose);
    }

    Ok(())
}

fn cols_to_expr(cols: Vec<&str>) -> Expr {
    cols.iter().map(|c| col(c)).collect::<Vec<_>>().into()
}

// ============================================================================
// CONVERT COMMAND
// ============================================================================

fn convert_files(
    input: PathBuf,
    output: PathBuf,
    from: Format,
    to: Format,
    delimiter: Option<char>,
    auto_delimiter: bool,
    _verbose: bool,
) -> Result<()> {
    let pb = create_progress_spinner(&format!("Converting {:?} → {:?}...", from, to));

    let input_delim = if matches!(from, Format::Csv) {
        if let Some(d) = delimiter {
            Some(d)
        } else if auto_delimiter {
            Some(detect_delimiter(&input)?)
        } else {
            Some(',')
        }
    } else {
        None
    };

    let output_delim = if matches!(to, Format::Csv) {
        delimiter.or(Some(','))
    } else {
        None
    };

    let mut df = load_dataframe(&input, input_delim, false)?;
    save_dataframe(&mut df, &output, Some(to), output_delim)?;

    pb.finish_with_message(format!("✅ Conversion complete. Output: {:?}", output));
    Ok(())
}

// ============================================================================
// QUERY COMMAND
// ============================================================================

fn query_data(
    input: PathBuf,
    output: PathBuf,
    sql: String,
    delimiter: Option<char>,
    auto_delimiter: bool,
    verbose: bool,
) -> Result<()> {
    let pb = create_progress_spinner("Executing SQL query...");

    let delim = if delimiter.is_some() {
        delimiter
    } else if auto_delimiter {
        Some(detect_delimiter(&input)?)
    } else {
        Some(',')
    };

    let df = load_dataframe(&input, delim, false)?;

    if verbose {
        println!("  Query: {}", sql);
    }

    // Create SQL context
    let mut ctx = SQLContext::new();
    ctx.register("self", df.lazy());

    let mut result_df = ctx
        .execute(&sql)
        .map_err(|e| anyhow::anyhow!("SQL execution error: {}", e))?
        .collect()
        .map_err(|e| anyhow::anyhow!("SQL collect error: {}", e))?;

    // Infer output format from extension (default CSV)
    save_dataframe(&mut result_df, &output, None, delim)?;

    pb.finish_with_message(format!("✅ Query complete. Output: {:?}", output));
    Ok(())
}

// ============================================================================
// PROFILE COMMAND
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
struct ProfileReport {
    file: String,
    row_count: usize,
    column_count: usize,
    columns: HashMap<String, ColumnProfile>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ColumnProfile {
    dtype: String,
    null_count: usize,
    null_percentage: f64,
    unique_count: Option<usize>,
    sample_values: Vec<String>,
}

fn profile_data(
    input: PathBuf,
    output: Option<PathBuf>,
    delimiter: Option<char>,
    auto_delimiter: bool,
    _verbose: bool,
) -> Result<()> {
    let pb = create_progress_spinner("Profiling data...");

    let delim = if delimiter.is_some() {
        delimiter
    } else if auto_delimiter {
        Some(detect_delimiter(&input)?)
    } else {
        Some(',')
    };

    let df = load_dataframe(&input, delim, false)?;

    let row_count = df.height();
    let column_count = df.width();

    let mut columns = HashMap::new();

    for col_name in df.get_column_names() {
        let series = df.column(col_name)?;
        let dtype = format!("{:?}", series.dtype());
        let null_count = series.null_count();
        let null_percentage = if row_count == 0 {
            0.0
        } else {
            (null_count as f64 / row_count as f64) * 100.0
        };

        // Get unique count (can be expensive for large datasets)
        let unique_count = if row_count < 100_000 {
            series.n_unique().ok()
        } else {
            None
        };

        // Get sample values (first 5 non-null)
        let sample_values: Vec<String> = series
            .head(Some(10))
            .iter()
            .filter_map(|v| {
                if v.is_null() {
                    None
                } else {
                    Some(format!("{}", v))
                }
            })
            .take(5)
            .collect();

        columns.insert(
            col_name.to_string(),
            ColumnProfile {
                dtype,
                null_count,
                null_percentage,
                unique_count,
                sample_values,
            },
        );
    }

    let report = ProfileReport {
        file: input.to_string_lossy().to_string(),
        row_count,
        column_count,
        columns,
    };

    pb.finish_and_clear();

    if let Some(out_path) = output {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&out_path, json)?;
        println!("✅ Profile complete. Report: {:?}", out_path);
    } else {
        let json = serde_json::to_string_pretty(&report)?;
        println!("{}", json);
    }

    Ok(())
}

// ============================================================================
// VALIDATE COMMAND
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
struct ValidationReport {
    file: String,
    valid: bool,
    row_count: usize,
    issues: Vec<ValidationIssue>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ValidationIssue {
    severity: String,
    column: Option<String>,
    message: String,
    affected_rows: Option<usize>,
}

fn validate_data(
    input: PathBuf,
    schema: Option<PathBuf>,
    report_path: PathBuf,
    delimiter: Option<char>,
    auto_delimiter: bool,
    verbose: bool,
) -> Result<()> {
    let pb = create_progress_spinner("Validating data...");

    let delim = if delimiter.is_some() {
        delimiter
    } else if auto_delimiter {
        Some(detect_delimiter(&input)?)
    } else {
        Some(',')
    };

    let df = load_dataframe(&input, delim, false)?;
    let row_count = df.height();

    let mut issues = Vec::new();

    // Basic validation: check for null values
    for col_name in df.get_column_names() {
        let series = df.column(col_name)?;
        let null_count = series.null_count();

        if null_count > 0 {
            issues.push(ValidationIssue {
                severity: "warning".to_string(),
                column: Some(col_name.to_string()),
                message: format!("Column has {} null values", null_count),
                affected_rows: Some(null_count),
            });
        }
    }

    // If schema provided, load and validate against it
    if let Some(schema_path) = schema {
        if verbose {
            println!("  Validating against schema: {:?}", schema_path);
        }
        // Schema validation would go here
        // For now, just note that schema was provided
        issues.push(ValidationIssue {
            severity: "info".to_string(),
            column: None,
            message: "Custom schema validation not yet implemented".to_string(),
            affected_rows: None,
        });
    }

    let valid = !issues.iter().any(|i| i.severity == "error");

    let report = ValidationReport {
        file: input.to_string_lossy().to_string(),
        valid,
        row_count,
        issues,
    };

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&report_path, json)?;

    pb.finish_with_message(format!(
        "✅ Validation complete. Report: {:?}",
        report_path
    ));

    if !valid {
        println!("⚠️  Validation found errors");
    }

    Ok(())
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();
    let verbose = cli.verbose;

    let result = match cli.command {
        Commands::Transform {
            input,
            output,
            columns,
            filter,
            drop_empty,
            delimiter,
            auto_delimiter,
            sample,
            stats,
            dry_run,
        } => {
            let filters = parse_filters(&filter)?;
            transform_csv(
                input,
                output,
                columns,
                filters,
                drop_empty,
                delimiter,
                auto_delimiter,
                sample,
                stats,
                dry_run,
                verbose,
            )
        }

        Commands::Convert {
            input,
            output,
            from,
            to,
            delimiter,
            auto_delimiter,
        } => convert_files(input, output, from, to, delimiter, auto_delimiter, verbose),

        Commands::Query {
            input,
            output,
            sql,
            delimiter,
            auto_delimiter,
        } => query_data(input, output, sql, delimiter, auto_delimiter, verbose),

        Commands::Profile {
            input,
            output,
            delimiter,
            auto_delimiter,
        } => profile_data(input, output, delimiter, auto_delimiter, verbose),

        Commands::Validate {
            input,
            schema,
            report,
            delimiter,
            auto_delimiter,
        } => validate_data(input, schema, report, delimiter, auto_delimiter, verbose),
    };

    if let Err(e) = result {
        eprintln!("❌ Error: {}", e);
        if verbose {
            eprintln!("\nDebug info: {:?}", e);
        }
        std::process::exit(1);
    }

    Ok(())
}
