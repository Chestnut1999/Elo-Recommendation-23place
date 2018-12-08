-- feat_no 904
-- #========================================================================
-- # Unix Date
-- #========================================================================
SELECT
  card_id
  ,unixdate_max
  ,unixdate_min
  ,unixdate_pdp
  ,unixdate_mean
  ,unixdate_std
  ,unixdate_mean - unixdate_median as unixdate_mean_diff_median
  ,unixdate_per75 - unixdate_median as unixdate_1percentile
  ,unixdate_median
  ,unixdate_per75
  ,unixdate_per25
FROM (
select
   card_id,
   max(UNIX_DATE(pdate)) as unixdate_max,
   min(UNIX_DATE(pdate)) as unixdate_min,
   max(UNIX_DATE(pdate)) - min(UNIX_DATE(pdate)) as unixdate_pdp,
   avg(UNIX_DATE(pdate)) as unixdate_mean,
   STDDEV_SAMP(UNIX_DATE(pdate)) as unixdate_std,
   max(unixdate_median) as unixdate_median,
   max(unixdate_per75) as unixdate_per75,
   max(unixdate_per25) as unixdate_per25
from (
SELECT
  card_id,
  cast(purchase_date as date) as pdate,
  PERCENTILE_CONT(UNIX_DATE( cast(purchase_date as date) ), 0.5) OVER(PARTITION BY card_id) as unixdate_median,
  PERCENTILE_CONT(UNIX_DATE( cast(purchase_date as date) ), 0.25) OVER(PARTITION BY card_id) as unixdate_per75,
  PERCENTILE_CONT(UNIX_DATE( cast(purchase_date as date) ), 0.75) OVER(PARTITION BY card_id) as unixdate_per25
from `hori.elo_historical`
where
  -- authorized_flag is True
  authorized_flag is False
)
group by
  card_id
)
;
