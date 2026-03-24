from google.cloud import bigquery
 
def main():
    client = bigquery.Client(project='gdelt-stock-sentiment-analysis')
    
    clean_tone_query = """
    CREATE OR REPLACE TABLE `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data_clean` AS
    WITH filled_prices AS (
        SELECT 
            event_date,
            company,
            ticker,
            daily_exposure_count,
            daily_avg_tone,
            -- Forward fill stock prices using LAST_VALUE window function
            LAST_VALUE(Open IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Open,
            LAST_VALUE(High IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as High,
            LAST_VALUE(Low IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Low,
            LAST_VALUE(Close IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Close,
            LAST_VALUE(Volume IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Volume
        FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data`
    )
    SELECT 
        event_date,
        company,
        ticker,
        daily_exposure_count,
        daily_avg_tone,
        Open,
        High,
        Low,
        Close,
        Volume,
        LEAD(Close) OVER (PARTITION BY ticker ORDER BY event_date) as next_day_close,
        SAFE_DIVIDE(Close - LAG(Close) OVER (PARTITION BY ticker ORDER BY event_date), 
                    LAG(Close) OVER (PARTITION BY ticker ORDER BY event_date)) * 100 as daily_return_pct,
        EXTRACT(DAYOFWEEK FROM event_date) as day_of_week
    FROM filled_prices
    WHERE Close IS NOT NULL
    ORDER BY ticker, event_date
    """
    
    client.query(clean_tone_query).result()

    
    clean_themes_query = """
    CREATE OR REPLACE TABLE `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_with_prices_clean` AS
    WITH filled_prices AS (
        SELECT 
            event_date,
            company,
            ticker,
            theme_category,
            exact_theme,
            daily_theme_mentions,
            daily_theme_avg_tone,
            LAST_VALUE(Open IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Open,
            LAST_VALUE(High IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as High,
            LAST_VALUE(Low IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Low,
            LAST_VALUE(Close IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Close,
            LAST_VALUE(Volume IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Volume
        FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_with_prices`
    )
    SELECT 
        event_date,
        company,
        ticker,
        theme_category,
        exact_theme,
        daily_theme_mentions,
        daily_theme_avg_tone,
        Open,
        High,
        Low,
        Close,
        Volume,
        LEAD(Close) OVER (PARTITION BY ticker ORDER BY event_date) as next_day_close
    FROM filled_prices
    WHERE Close IS NOT NULL
    ORDER BY ticker, event_date
    """
    
    client.query(clean_themes_query).result()
    
    # Export cleaned data
    export_clean_tone = """
    EXPORT DATA OPTIONS(
      uri='gs://og-gdelt-main-data-dev/cleaned_data/combined_data_clean_*.csv',
      format='CSV',
      overwrite=true,
      header=true
    ) AS
    SELECT * FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data_clean`
    """
    client.query(export_clean_tone).result()

    export_clean_themes = """
    EXPORT DATA OPTIONS(
        uri='gs://og-gdelt-main-data-dev/cleaned_data/themes_with_prices_clean_*.csv',
        format='CSV',
        overwrite=true,
        header=true
    ) AS
    SELECT * FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_with_prices_clean`
    """
    client.query(export_clean_themes).result()
    
    #Summary
    summary_query = """
    SELECT 
        'combined_data_clean' as table_name,
        COUNT(*) as total_rows,
        COUNTIF(next_day_close IS NULL) as missing_next_day_close,
        ROUND(AVG(daily_avg_tone), 2) as avg_tone,
        ROUND(STDDEV(daily_avg_tone), 2) as stddev_tone
    FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data_clean`
    """
    
    result = client.query(summary_query).result()
    for row in result:
        print(f"Table: {row.table_name}")
        print(f"  Total rows: {row.total_rows}")
        print(f"  Missing next_day_close: {row.missing_next_day_close}")
        print(f"  Avg tone: {row.avg_tone}")
        print(f"  Stddev tone: {row.stddev_tone}")
 
if __name__ == '__main__':
    main()