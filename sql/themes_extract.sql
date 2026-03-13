WITH filtered AS (
  -- Filter the raw data down to just your target companies
  SELECT 
    DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as event_date,
    V2Themes,
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
    AND V2Themes IS NOT NULL
    AND (
      REGEXP_CONTAINS(V2Organizations, r'(?i)\bamazon\b|\bamzn\b') OR
      REGEXP_CONTAINS(V2Organizations, r'(?i)\bpfizer\b|\bpfe\b') OR
      REGEXP_CONTAINS(V2Organizations, r'(?i)\baramco\b|\bsaudi aramco\b|\bsaudi arabian oil\b')
    )
)

-- Unnest the themes and aggregate by day, company, and theme
SELECT 
  event_date,
  company,
  SPLIT(individual_theme, '_')[OFFSET(0)] as theme_category,
  individual_theme as exact_theme,
  COUNT(DISTINCT DocumentIdentifier) as daily_theme_mentions,
  AVG(primary_tone) as daily_theme_avg_tone
FROM 
  filtered,
  UNNEST(SPLIT(V2Themes, ';')) as individual_theme
WHERE 
  individual_theme != ''
GROUP BY 
  event_date,
  company,
  theme_category,
  exact_theme
;