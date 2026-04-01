SELECT 
  DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as event_date,
  '{company_name}' as company,
  COUNT(DISTINCT DocumentIdentifier) as daily_exposure_count,
  AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) as daily_avg_tone
FROM 
  `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE 
  _PARTITIONDATE BETWEEN '{start_date}' AND '{end_date}'
  AND REGEXP_CONTAINS(V2Organizations, r'(?i)\b{company_regex}\b')
GROUP BY event_date, company
ORDER BY event_date ASC