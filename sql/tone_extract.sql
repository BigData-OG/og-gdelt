WITH filtered AS (
  SELECT 
    DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as event_date,
    CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) as primary_tone,
    DocumentIdentifier,
    CASE 
      WHEN REGEXP_CONTAINS(V2Organizations, r'(?i)\bamazon\b|\bamzn\b') THEN 'Amazon'
      WHEN REGEXP_CONTAINS(V2Organizations, r'(?i)\bpfizer\b|\bpfe\b') THEN 'Pfizer'
      WHEN REGEXP_CONTAINS(V2Organizations, r'(?i)\baramco\b|\bsaudi aramco\b|\bsaudi arabian oil\b') THEN 'Aramco'
    END as company
  FROM 
    `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE 
    _PARTITIONDATE BETWEEN '2020-01-01' AND '2025-12-31'
    AND (
      REGEXP_CONTAINS(V2Organizations, r'(?i)\bamazon\b|\bamzn\b') OR
      REGEXP_CONTAINS(V2Organizations, r'(?i)\bpfizer\b|\bpfe\b') OR
      REGEXP_CONTAINS(V2Organizations, r'(?i)\baramco\b|\bsaudi aramco\b|\bsaudi arabian oil\b')
    )
)

SELECT 
  event_date,
  company,
  COUNT(DISTINCT DocumentIdentifier) as daily_exposure_count,
  AVG(primary_tone) as daily_avg_tone
FROM 
  filtered
GROUP BY 
  event_date, 
  company
ORDER BY 
  event_date DESC, 
  company