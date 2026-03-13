from google.cloud import bigquery

client = bigquery.Client(project='gdelt-stock-sentiment-analysis')

# Run query and save to GCS
query = """

"""

job = client.query(query)
job.result()