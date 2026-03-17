from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("GDELT-Market-Alignment").getOrCreate()

# loading data locally 
# TODO: change
market_df = spark.read.csv("market_data_ready_for_join.csv", header=True, inferSchema=True)
gdelt_df = spark.read.csv("gdelt_sample.csv", header=True, inferSchema=True)

# split V2Tone and take the first value in cell
gdelt_processed = gdelt_df.withColumn(
    "Sentiment", 
    F.split(F.col("V2Tone"), ",").getItem(0).cast("float")
)

# aggregate news
daily_news = gdelt_processed.groupBy("event_date", "Ticker").agg(
    F.avg("Sentiment").alias("Avg_Sentiment"),
    F.count("DocumentIdentifier").alias("Article_Count")
)

# join news and market data

aligned_df = market_df.join(
    daily_news, 
    (market_df.Date == daily_news.event_date) & (market_df.Ticker == daily_news.Ticker), 
    "left"
)

# drop duplicates
final_data = aligned_df.drop("event_date").drop(daily_news.Ticker)

# preview only
final_data.sort("Date", "Ticker").show()

# save final file
final_data.coalesce(1).write.mode("overwrite").csv("final_aligned_prototype", header=True)

print("done")