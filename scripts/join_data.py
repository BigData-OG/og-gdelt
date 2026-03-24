from google.cloud import bigquery
import pandas as pd

def load_csv_to_bigquery(client, gcs_pattern, dataset_id, table_id):
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition='WRITE_TRUNCATE'
    )
    
    load_job = client.load_table_from_uri(
        gcs_pattern,
        table_ref,
        job_config=job_config
    )
    load_job.result()

def main():
    client = bigquery.Client(project='gdelt-stock-sentiment-analysis')
    
    load_csv_to_bigquery(
        client,
        'gs://og-gdelt-main-data-dev/gdelt_raw/themes_data*.csv',
        'gdelt_analysis',
        'themes_raw'
    )
    
    load_csv_to_bigquery(
        client,
        'gs://og-gdelt-main-data-dev/gdelt_raw/tone_exposure_data_*.csv',
        'gdelt_analysis',
        'tone_exposure_raw'
    )
    
    load_csv_to_bigquery(
        client,
        'gs://og-gdelt-main-data-dev/yfinance_raw/market_data_raw.csv',
        'gdelt_analysis',
        'stock_prices'
    )
    
    # join in BigQuery
    tone_exposure_query = """
    CREATE OR REPLACE TABLE `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data` AS
    SELECT 
        t.event_date,
        t.company,
        CASE 
            WHEN t.company = 'Amazon' THEN 'AMZN'
            WHEN t.company = 'Pfizer' THEN 'PFE'
            WHEN t.company = 'Aramco' THEN '2222.SR'
        END as ticker,
        t.daily_exposure_count,
        t.daily_avg_tone,
        s.Open,
        s.High,
        s.Low,
        s.Close,
        s.Volume
    FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.tone_exposure_raw` t
    LEFT JOIN `gdelt-stock-sentiment-analysis.gdelt_analysis.stock_prices` s
        ON t.event_date = s.Date
        AND CASE 
            WHEN t.company = 'Amazon' THEN 'AMZN'
            WHEN t.company = 'Pfizer' THEN 'PFE'
            WHEN t.company = 'Aramco' THEN '2222.SR'
        END = s.Ticker
    """
    client.query(tone_exposure_query).result()

    themes_join_query = """
    CREATE OR REPLACE TABLE `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_with_prices` AS
    SELECT 
        th.event_date,
        th.company,
        CASE 
            WHEN th.company = 'Amazon' THEN 'AMZN'
            WHEN th.company = 'Pfizer' THEN 'PFE'
            WHEN th.company = 'Aramco' THEN '2222.SR'
        END as ticker,
        th.theme_category,
        th.exact_theme,
        th.daily_theme_mentions,
        th.daily_theme_avg_tone,
        s.Open,
        s.High,
        s.Low,
        s.Close,
        s.Volume
    FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_raw` th
    LEFT JOIN `gdelt-stock-sentiment-analysis.gdelt_analysis.stock_prices` s
        ON th.event_date = s.Date
        AND CASE 
            WHEN th.company = 'Amazon' THEN 'AMZN'
            WHEN th.company = 'Pfizer' THEN 'PFE'
            WHEN th.company = 'Aramco' THEN '2222.SR'
        END = s.Ticker
    """
    client.query(themes_join_query).result()
    
    # Export
    export_tone_query = """
    EXPORT DATA OPTIONS(
      uri='gs://og-gdelt-main-data-dev/analysis_ready/combined_data_*.csv',
      format='CSV',
      overwrite=true,
      header=true
    ) AS
    SELECT * FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data`
    """
    client.query(export_tone_query).result()
    
    export_themes_query = """
    EXPORT DATA OPTIONS(
      uri='gs://og-gdelt-main-data-dev/analysis_ready/themes_with_prices_*.csv',
      format='CSV',
      overwrite=true,
      header=true
    ) AS
    SELECT * FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_with_prices`
    """
    client.query(export_themes_query).result()

if __name__ == '__main__':
    main()