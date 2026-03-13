from google.cloud import bigquery
from pathlib import Path
import os

def read_sql_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def execute_query_to_gcs(client, query, output_uri):
    export_query = f"""
    EXPORT DATA OPTIONS(
      uri='{output_uri}',
      format='CSV',
      overwrite=true,
      header=true
    ) AS
    {query}
    """
    
    job = client.query(export_query)
    result = job.result()
    return result

def main():
    client = bigquery.Client(project='gdelt-stock-sentiment-analysis')
    
    sql_dir = Path(__file__).parent.parent / 'sql'
    gcs_bucket = 'gs://og-gdelt-main-data-dev'
    
    # Define queries to run
    queries = [
        {
            'name': 'tone_extract',
            'sql_file': sql_dir / 'tone_extract.sql',
            'output_uri': f'{gcs_bucket}/gdelt_raw/tone&exposure_data.csv'
        },
        {
            'name': 'themes_extract',
            'sql_file': sql_dir / 'themes_extract.sql',
            'output_uri': f'{gcs_bucket}/gdelt_raw/themes_data.csv'
        }
    ]
    
    for query_config in queries:
        sql_query = read_sql_file(query_config['sql_file'])
        
        try:
            execute_query_to_gcs(
                client=client,
                query=sql_query,
                output_uri=query_config['output_uri']
            )
            print(f"✓ {query_config['name']} completed successfully")
        except Exception as e:
            print(f"✗ Error executing {query_config['name']}: {e}")

if __name__ == '__main__':
    main()