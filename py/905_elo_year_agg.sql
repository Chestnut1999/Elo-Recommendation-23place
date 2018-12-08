-- feat_no 905
-- #========================================================================
-- # Year aggregation
-- #========================================================================

# Create Table
WITH
  base as (
    SELECT
      card_id,
      purchase_date,
      purchase_amount,
      installments,
      merchant_category_id,
      merchant_id,
      authorized_flag,
      EXTRACT(year FROM purchase_date) as year,
      EXTRACT(month FROM purchase_date) as month,
      EXTRACT(dayofweek FROM purchase_date) as dow,
      EXTRACT(hour FROM purchase_date) as hour
    from `hori.elo_historical`
  )
  ,
  t_latest_year as (
    select
      card_id,
      max(year) as latest_year
    FROM base
    GROUP BY
      card_id
  )
  ,
  t_latest_month_number as (
    SELECT
    distinct
      year,
      month,
      DENSE_RANK()over(ORDER BY year desc, month desc) as latest_month_no
    FROM base
  )
  ,
  tmp_yyyymm_no as (
    SELECT
      distinct
      t1.card_id,
      t1.year,
      t1.month,
      latest_month_no
    FROM
      base as t1
    INNER JOIN
      t_latest_month_number t2
      on t1.year = t2.year and t1.month = t2.month
  )
  ,
  yyyymm_no as (
    SELECT
      t1.card_id,
      year,
      month,
      latest_month_no - min_latest_month_no + 1 as latest_month_no
    FROM
      tmp_yyyymm_no as t1
    INNER JOIN (
      SELECT
        card_id,
        min(latest_month_no) as min_latest_month_no
      FROM
        tmp_yyyymm_no
      GROUP BY
        card_id
    ) as t2
      on t1.card_id = t2.card_id
  )
  ,
  time_detail as (
  SELECT
    t1.card_id,
    purchase_amount,
    installments,
    purchase_date,
    merchant_category_id,
    merchant_id,
    CASE
      WHEN (hour  BETWEEN 22 AND 24 ) OR (hour  BETWEEN 0 AND 4 ) THEN 'night'
      WHEN (hour  BETWEEN 18 AND 22 ) THEN 'midnight'
      WHEN (hour  BETWEEN 14 AND 17 ) THEN 'afternoon'
      WHEN (hour  BETWEEN 11 AND 13 ) THEN 'noon'
      WHEN (hour  BETWEEN 5 AND 10 ) THEN 'morning'
      ELSE 'other'
    END as timezone,
    CONCAT(
    CAST(dow AS STRING), '_',
    CASE
      WHEN (hour  BETWEEN 22 AND 24 ) OR (hour  BETWEEN 0 AND 4 ) THEN 'night'
      WHEN (hour  BETWEEN 18 AND 22 ) THEN 'midnight'
      WHEN (hour  BETWEEN 14 AND 17 ) THEN 'afternoon'
      WHEN (hour  BETWEEN 11 AND 13 ) THEN 'noon'
      WHEN (hour  BETWEEN 5 AND 10 ) THEN 'morning'
      ELSE 'other'
    END) as dow_timezone,
    t1.year,
    t1.month,
    dow,
    hour,
    latest_year,
    latest_month_no,
    authorized_flag
  FROM base as t1
  INNER JOIN t_latest_year as t2
    on t1.card_id = t2.card_id
  INNER JOIN yyyymm_no as t3
    on t1.card_id = t3.card_id and t1.year = t3.year and t1.month = t3.month
)

SELECT
  *
FROM time_detail
;


# year agg
WITH
  year_agg as (
  SELECT
    card_id,
    year,
    count(1) as transactions_cnt__card_id_year,
    sum(purchase_amount) as purchase_amount_sum__card_id_year,
    avg(purchase_amount) as purchase_amount_mean__card_id_year,
    max(purchase_amount) as purchase_amount_max__card_id_year,
    min(purchase_amount) as purchase_amount_min__card_id_year,
    STDDEV_SAMP(purchase_amount) as purchase_amount_std__card_id_year,

    sum(installments) as installments_sum__card_id_year,
    avg(installments) as installments_mean__card_id_year,
    max(installments) as installments_max__card_id_year,
    min(installments) as installments_min__card_id_year,
    STDDEV_SAMP(installments) as installments_std__card_id_year,
    count(distinct installments) as installments_nuq__card_id_year,

    count(distinct cast(purchase_date as date)) as date_nuq__card_id_year,
    count(distinct dow) as dow_nuq__card_id_year,
    count(distinct timezone) as timezone_nuq__card_id_year,
    count(distinct dow_timezone) as dow_timezone_nuq__card_id_year,
    count(distinct month) as month_nuq__card_id_year,
    count(distinct merchant_category_id) as merchant_category_id_nuq__card_id_year,
    count(distinct merchant_id) as merchant_id_nuq__card_id_year

  FROM
    hori.historical_time_detail
  where
    authorized_flag is false
  GROUP BY
    card_id,
    year
)
SELECT
  *
FROM year_agg
;
