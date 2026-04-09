from google.cloud import bigquery, storage
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Extracts GDELT tone data and yFinance stock prices.
    Processes and combines them into training-ready dataset.
    """

    def __init__(self, bucket, project_id):
        self.bq_client = bigquery.Client(project=project_id)
        self.gcs_client = storage.Client(project=project_id)
        self.bucket_name = bucket
        self.sql_dir = Path(__file__).parent.parent / 'sql'

    # Larry can call this
    def extract_company_data(self, company_name: str, ticker: str, years: int = 5) -> dict:
        """
        Main extraction method - extracts tone + stock data and processes it.
        
        Args:
            company_name: Company name (e.g., "Tesla")
            ticker: Stock ticker (e.g., "TSLA")
            years: Years of historical data (default: 5)
            
        Returns:
            dict with paths, counts, and metadata
        """
        logger.info(f"Starting data extraction for {company_name} ({ticker})")

        # date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        # Get standardized GCS paths
        paths = self._get_gcs_paths(ticker)

        try:
            # Extract GDELT tone data
            tone_count = self._extract_gdelt_tone(
                company_name, ticker, start_date, end_date, paths['gdelt_tone']
            )
            logger.info(f"✓ Extracted {tone_count} rows of tone data")

            # Extract yFinance stock data
            yf_count = self._extract_yfinance(
                ticker, start_date, end_date, paths['yfinance']
            )
            logger.info(f"✓ Extracted {yf_count} rows of stock data")

            # join + clean + feature engineering
            processed_count = self._process_data(ticker, paths)
            logger.info(f"✓ Processed {processed_count} rows of training data")

            return {
                'status': 'success',
                'ticker': ticker,
                'company_name': company_name,
                'paths': paths,
                'data_range': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                },
                'row_counts': {
                    'gdelt_tone': tone_count,
                    'yfinance': yf_count,
                    'processed': processed_count
                }
            }

        except Exception as e:
            logger.error(f"Error extracting data for {ticker}: {e}")
            raise

    def get_latest_features(self, company_name: str, ticker: str) -> dict:
        """
        Get only the latest day's features for prediction (fast).
        """
        logger.info(f"Getting latest features for {company_name} ({ticker})")

        # Get today's data only (1 day)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # 7 days buffer

        try:
            gdelt_query = f"""
            SELECT 
                DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as event_date,
                COUNT(DISTINCT DocumentIdentifier) as daily_exposure_count,
                AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) as daily_avg_tone
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`
                WHERE _PARTITIONDATE BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
              AND REGEXP_CONTAINS(V2Organizations, r'(?i)\\b{company_name.lower()}|{ticker.lower()}\\b')
            GROUP BY event_date
            ORDER BY event_date DESC
            LIMIT 1
            """

            gdelt_result = self.bq_client.query(gdelt_query).result()
            gdelt_row = list(gdelt_result)[0]

            # Get latest stock price from yFinance
            stock_data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )

            if stock_data.empty:
                raise ValueError(f"No recent stock data for {ticker}")

            # Get last row
            latest_stock = stock_data.iloc[-1]

            # calc features
            if len(stock_data) >= 2:
                prev_close = stock_data.iloc[-2]['Close']
                daily_return_pct = ((latest_stock['Close'] - prev_close) / prev_close) * 100
            else:
                daily_return_pct = 0.0

            # Build feature dict
            features = {
                'daily_exposure_count': float(gdelt_row.daily_exposure_count),
                'daily_avg_tone': float(gdelt_row.daily_avg_tone),
                'Open': float(latest_stock['Open']),
                'High': float(latest_stock['High']),
                'Low': float(latest_stock['Low']),
                'Close': float(latest_stock['Close']),
                'Volume': float(latest_stock['Volume']),
                'daily_return_pct': float(daily_return_pct),
                'day_of_week': int(stock_data.index[-1].dayofweek)
            }

            return {
                'status': 'success',
                'ticker': ticker,
                'company_name': company_name,
                'event_date': gdelt_row.event_date.strftime('%Y-%m-%d'),
                'features': features
            }

        except Exception as e:
            logger.error(f"Error getting latest features for {ticker}: {e}")
            raise

    # === helper functions, DONT CALL THE FUNCTIONS BELOW ===

    def _get_gcs_paths(self, ticker: str) -> dict:
        """Generate standardized GCS paths"""
        base = f"gs://{self.bucket_name}/companies/{ticker}"
        return {
            'gdelt_tone': f"{base}/gdelt_raw/tone_data*.csv",
            'yfinance': f"{base}/yfinance_raw/stock_prices.csv",
            'processed': f"{base}/processed/training_data*.csv"
        }

    def _extract_gdelt_tone(self, company_name: str, ticker: str,
                            start_date: datetime, end_date: datetime,
                            output_path: str) -> int:
        """Extract GDELT tone data"""

        # Read SQL script
        template_path = self.sql_dir / 'tone_extract.sql'
        with open(template_path, 'r') as f:
            template = f.read()
        print("IT WORKSSSSSSSSS")
        company_regex = f"{company_name.lower()}|{ticker.lower()}"

        # Fill in parameters
        query = template.format(
            company_name=company_name,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            company_regex=company_regex
        )

        # Export to GCS
        export_query = f"""
        EXPORT DATA OPTIONS(
          uri='{output_path}',
          format='CSV',
          overwrite=true,
          header=true
        ) AS
        {query}
        """

        job = self.bq_client.query(export_query)
        job.result()

        # Get row count
        count_query = f"SELECT COUNT(*) as cnt FROM ({query})"
        result = self.bq_client.query(count_query).result()
        return list(result)[0].cnt

    def _extract_yfinance(self, ticker: str, start_date: datetime,
                          end_date: datetime, output_path: str) -> int:
        """Extract yFinance stock price data"""

        # Download data
        raw_data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if raw_data.empty:
            raise ValueError(f"No stock data found for {ticker}")

        # Stack and format
        stacked = raw_data.stack(level=1, future_stack=True) if isinstance(raw_data.columns,
                                                                           pd.MultiIndex) else raw_data
        stacked = stacked.reset_index()

        # Rename columns
        col_names = stacked.columns.tolist()
        if 'Date' not in col_names:
            stacked.rename(columns={col_names[0]: 'Date'}, inplace=True)

        # Add Ticker column
        stacked['Ticker'] = ticker

        # Forward fill missing values (weekends/holidays)
        stacked = stacked.sort_values('Date').reset_index(drop=True)
        stacked = stacked.ffill()

        # Format Date
        stacked['Date'] = pd.to_datetime(stacked['Date'])
        stacked['Date'] = stacked['Date'].dt.strftime('%Y-%m-%d')

        # Reorder columns
        join_keys = ['Date', 'Ticker']
        price_cols = [c for c in stacked.columns if c not in join_keys]
        stacked = stacked[join_keys + sorted(price_cols)]

        # Upload to GCS
        bucket = self.gcs_client.bucket(self.bucket_name)
        blob_path = output_path.replace(f"gs://{self.bucket_name}/", "")
        blob = bucket.blob(blob_path)
        blob.upload_from_string(stacked.to_csv(index=False), content_type='text/csv')

        logger.info(f"yFinance data uploaded to {output_path}")
        return len(stacked)

    def _process_data(self, ticker: str, paths: dict) -> int:
        """
        Join tone data with stock prices and create training dataset.
        
        Steps:
        1. Load tone and stock data to BigQuery temp tables
        2. Join on event_date = Date
        3. Forward-fill missing stock prices
        4. Calculate next_day_close (target variable)
        5. Calculate daily_return_pct (feature)
        6. Add day_of_week (feature)
        7. Export to GCS
        """

        # Create temp dataset
        temp_dataset = f"temp_{ticker.replace('.', '_').replace('-', '_')}"
        dataset = bigquery.Dataset(f"{self.bq_client.project}.{temp_dataset}")
        dataset.location = "US"
        self.bq_client.create_dataset(dataset, exists_ok=True)

        try:
            # Load data to BigQuery
            self._load_to_bq(paths['gdelt_tone'], temp_dataset, 'tone_raw')
            self._load_to_bq(paths['yfinance'], temp_dataset, 'stock_prices')

            # Join and clean query
            query = f"""
            EXPORT DATA OPTIONS(
              uri='{paths['processed']}',
              format='CSV',
              overwrite=true,
              header=true
            ) AS
            WITH joined AS (
                SELECT 
                    t.event_date,
                    t.company,
                    '{ticker}' as ticker,
                    t.daily_exposure_count,
                    t.daily_avg_tone,
                    s.Open,
                    s.High,
                    s.Low,
                    s.Close,
                    s.Volume
                FROM `{self.bq_client.project}.{temp_dataset}.tone_raw` t
                LEFT JOIN `{self.bq_client.project}.{temp_dataset}.stock_prices` s
                    ON t.event_date = s.Date
                    AND s.Ticker = '{ticker}'
            ),
            filled AS (
                SELECT 
                    event_date,
                    company,
                    ticker,
                    daily_exposure_count,
                    daily_avg_tone,
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
                FROM joined
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
                ROUND((Close - LAG(Close) OVER (PARTITION BY ticker ORDER BY event_date)) 
                    / LAG(Close) OVER (PARTITION BY ticker ORDER BY event_date) * 100, 2) as daily_return_pct,
                EXTRACT(DAYOFWEEK FROM event_date) - 1 as day_of_week
            FROM filled
            WHERE Close IS NOT NULL
            ORDER BY event_date ASC
            """

            job = self.bq_client.query(query)
            job.result()

            # Get row count from temp table
            count_query = f"SELECT COUNT(*) as cnt FROM `{self.bq_client.project}.{temp_dataset}.tone_raw`"
            result = self.bq_client.query(count_query).result()
            row_count = list(result)[0].cnt

            return row_count

        finally:
            # Cleanup temp dataset
            self.bq_client.delete_dataset(temp_dataset, delete_contents=True, not_found_ok=True)

    def _load_to_bq(self, gcs_path: str, dataset: str, table: str):
        """Load CSV from GCS to BigQuery table"""
        table_ref = f"{self.bq_client.project}.{dataset}.{table}"

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
            write_disposition='WRITE_TRUNCATE'
        )

        load_job = self.bq_client.load_table_from_uri(
            gcs_path,
            table_ref,
            job_config=job_config
        )
        load_job.result()
        logger.info(f"Loaded {gcs_path} to {table_ref}")
