-- =================================
-- Create Table
-- =================================
with
  auth as(
    SELECT
    distinct
      card_id,
      purchase_date
    FROM `hori.elo_historical`
    WHERE authorized_flag IS TRUE
  )
  ,
  lag_lead as(
    select
      card_id,
      purchase_date,
      lag(purchase_date, 1)over(partition by card_id order by purchase_date ) lag_1,
      lag(purchase_date, 2)over(partition by card_id order by purchase_date ) lag_2,
      lag(purchase_date, 3)over(partition by card_id order by purchase_date ) lag_3,
      lead(purchase_date, 1)over(partition by card_id order by purchase_date) lead_1,
      lead(purchase_date, 2)over(partition by card_id order by purchase_date) lead_2,
      lead(purchase_date, 3)over(partition by card_id order by purchase_date) lead_3
    from auth
)
,
diff_days as(
  select
    card_id,
    purchase_date,
    date_diff(cast(purchase_date as date), cast(lag_1 as date), DAY) as diff_days_lag1,
    date_diff(cast(purchase_date as date), cast(lag_2 as date), DAY) as diff_days_lag2,
    date_diff(cast(purchase_date as date), cast(lag_3 as date), DAY) as diff_days_lag3,
    date_diff(cast(purchase_date as date), cast(lead_1 as date), DAY) as diff_days_lead1,
    date_diff(cast(purchase_date as date), cast(lead_2 as date), DAY) as diff_days_lead2,
    date_diff(cast(purchase_date as date), cast(lead_3 as date), DAY) as diff_days_lead3
  from lag_lead
)
,
ratio_days as(
  select
    card_id,
    purchase_date,
    diff_days_lag1 / (diff_days_lag2  + 0.0000001) as ratio_days_lag1_2,
    diff_days_lag1 / (diff_days_lag3  + 0.0000001) as ratio_days_lag1_3,
    diff_days_lag2 / (diff_days_lag3  + 0.0000001) as ratio_days_lag2_3,
    diff_days_lag1 / (diff_days_lead1 + 0.0000001) as ratio_days_lag1_lead1,
    diff_days_lag2 / (diff_days_lead2 + 0.0000001) as ratio_days_lag2_lead2,
    diff_days_lag3 / (diff_days_lead3 + 0.0000001) as ratio_days_lag3_lead3
  from diff_days
),
-- 前回と何日間隔空くことが多いか？これを横持ちにする
cnt_diff_days as(
  select
    card_id,
    diff_days_lag1,
    count(1) as cnt_diff_days_lag1
  from diff_days
  group by
    card_id,
    diff_days_lag1
)
,
-- historicalのlatestデータ点を基準に過去1~24か月前のポイントを取得する
-- これを元に集計期間を決める
his_max_date as(
  select
    card_id,
    DATE_ADD(cast(max_date as date), INTERVAL -30 DAY) as his_latest_30,
    DATE_ADD(cast(max_date as date), INTERVAL -60 DAY) as his_latest_60,
    DATE_ADD(cast(max_date as date), INTERVAL -90 DAY) as his_latest_90,
    DATE_ADD(cast(max_date as date), INTERVAL -120 DAY) as his_latest_120,
    DATE_ADD(cast(max_date as date), INTERVAL -150 DAY) as his_latest_150,
    DATE_ADD(cast(max_date as date), INTERVAL -180 DAY) as his_latest_180,
    DATE_ADD(cast(max_date as date), INTERVAL -360 DAY) as his_latest_360,
    DATE_ADD(cast(max_date as date), INTERVAL -540 DAY) as his_latest_540,
    DATE_ADD(cast(max_date as date), INTERVAL -720 DAY) as his_latest_720
  from (
    select
      card_id,
      max(purchase_date) as max_date
    from auth
    group by
      card_id
  ) as t1
)
,
-- historicalのlatestデータ点を基準に過去1~24か月前のポイントを取得する
-- これを元に集計期間を決める
his_min_date as(
  select
    card_id,
    DATE_ADD(cast(min_date as date), INTERVAL 30 DAY) as his_first_30,
    DATE_ADD(cast(min_date as date), INTERVAL 60 DAY) as his_first_60,
    DATE_ADD(cast(min_date as date), INTERVAL 90 DAY) as his_first_90,
    DATE_ADD(cast(min_date as date), INTERVAL 120 DAY) as his_first_120
  from (
    select
      card_id,
      min(purchase_date) as min_date
    from auth
    group by
      card_id
  ) as t1
)
,
-- historicalのlatestデータ点を基準に過去1~24か月前のポイントを取得する
-- これを元に集計期間を決める
new_max_date as(
  select
    card_id,
    DATE_ADD(cast(max_date as date), INTERVAL -30 DAY) as new_latest_30,
    DATE_ADD(cast(max_date as date), INTERVAL -60 DAY) as new_latest_60,
    DATE_ADD(cast(max_date as date), INTERVAL -90 DAY) as new_latest_90,
    DATE_ADD(cast(max_date as date), INTERVAL -120 DAY) as new_latest_120,
    DATE_ADD(cast(max_date as date), INTERVAL -150 DAY) as new_latest_150,
    DATE_ADD(cast(max_date as date), INTERVAL -180 DAY) as new_latest_180,
    DATE_ADD(cast(max_date as date), INTERVAL -360 DAY) as new_latest_360,
    DATE_ADD(cast(max_date as date), INTERVAL -540 DAY) as new_latest_540,
    DATE_ADD(cast(max_date as date), INTERVAL -720 DAY) as new_latest_720
  from (
    select
      card_id,
      max(purchase_date) as max_date
    from hori.new
    group by
      card_id
  ) as t1
)
,
-- historicalのlatestデータ点を基準に過去1~24か月前のポイントを取得する
-- これを元に集計期間を決める
new_min_date as(
  select
    card_id,
    DATE_ADD(cast(min_date as date), INTERVAL 30 DAY) as new_first_30,
    DATE_ADD(cast(min_date as date), INTERVAL 60 DAY) as new_first_60,
    DATE_ADD(cast(min_date as date), INTERVAL 90 DAY) as new_first_90,
    DATE_ADD(cast(min_date as date), INTERVAL 0 DAY) as new_first_120
  from (
    select
      card_id,
      min(purchase_date) as min_date
    from hori.new
    group by
      card_id
  ) as t1
)
,
join_term as (
  SELECT
    t1.card_id,
    cast(t1.purchase_date as date) as p_date,
    -- diff
    diff_days_lag1,
    diff_days_lag2,
    diff_days_lag3,
    diff_days_lead1,
    diff_days_lead2,
    diff_days_lead3,
    -- ratio
    ratio_days_lag1_2,
    ratio_days_lag1_3,
    ratio_days_lag2_3,
    ratio_days_lag1_lead1,
    ratio_days_lag2_lead2,
    ratio_days_lag3_lead3,
    -- latest
    his_latest_30,
    his_latest_60,
    his_latest_90,
    his_latest_120,
    his_latest_150,
    his_latest_180,
    his_latest_360,
    his_latest_540,
    his_latest_720,
    new_latest_30,
    new_latest_60,
    new_latest_90,
    new_latest_120,
    new_latest_150,
    new_latest_180,
    new_latest_360,
    new_latest_540,
    new_latest_720,
    -- first
    his_first_30,
    his_first_60,
    his_first_90,
    his_first_120,
    new_first_30,
    new_first_60,
    new_first_90,
    new_first_120
  FROM diff_days as t1
  INNER JOIN his_max_date as t2
    ON t1.card_id = t2.card_id
  INNER JOIN his_min_date as t3
    ON t1.card_id = t3.card_id
  INNER JOIN new_max_date as t4
    ON t1.card_id = t4.card_id
  INNER JOIN new_min_date as t5
    ON t1.card_id = t5.card_id
  INNER JOIN ratio_days as t6
    ON t1.card_id = t6.card_id and t1.purchase_date = t6.purchase_date
)
SELECT
  *
FROM join_term
;


-- =================================
-- Aggregation
-- =================================

WITH
his_latest_30 as (
  SELECT
    card_id,
    -- mean
    avg(diff_days_lag1)        As his_latest30_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest30_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest30_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest30_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest30_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest30_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest30_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest30_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest30_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest30_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest30_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest30_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest30_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest30_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest30_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest30_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest30_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest30_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest30_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest30_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest30_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest30_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest30_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest30_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest30_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest30_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest30_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest30_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest30_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest30_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest30_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest30_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest30_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest30_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest30_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest30_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest30_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest30_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest30_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest30_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest30_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest30_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest30_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest30_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest30_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest30_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest30_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest30_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest30_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest30_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest30_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest30_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest30_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest30_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest30_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest30_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest30_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest30_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest30_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest30_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_30
  GROUP BY
    card_id
)
,
his_latest_60 as (
  SELECT
    card_id as id_l60,
    -- mean
    avg(diff_days_lag1)        As his_latest60_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest60_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest60_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest60_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest60_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest60_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest60_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest60_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest60_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest60_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest60_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest60_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest60_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest60_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest60_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest60_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest60_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest60_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest60_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest60_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest60_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest60_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest60_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest60_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest60_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest60_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest60_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest60_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest60_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest60_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest60_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest60_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest60_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest60_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest60_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest60_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest60_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest60_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest60_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest60_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest60_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest60_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest60_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest60_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest60_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest60_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest60_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest60_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest60_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest60_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest60_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest60_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest60_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest60_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest60_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest60_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest60_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest60_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest60_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest60_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_60
  GROUP BY
    card_id
)
,
his_latest_90 as (
  SELECT
    card_id as id_l90,
    -- mean
    avg(diff_days_lag1)        As his_latest90_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest90_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest90_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest90_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest90_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest90_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest90_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest90_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest90_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest90_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest90_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest90_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest90_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest90_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest90_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest90_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest90_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest90_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest90_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest90_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest90_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest90_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest90_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest90_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest90_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest90_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest90_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest90_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest90_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest90_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest90_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest90_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest90_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest90_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest90_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest90_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest90_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest90_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest90_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest90_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest90_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest90_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest90_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest90_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest90_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest90_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest90_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest90_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest90_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest90_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest90_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest90_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest90_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest90_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest90_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest90_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest90_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest90_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest90_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest90_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_90
  GROUP BY
    card_id
)
,
his_latest_120 as (
  SELECT
    card_id as id_l120,
    -- mean
    avg(diff_days_lag1)        As his_latest120_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest120_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest120_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest120_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest120_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest120_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest120_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest120_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest120_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest120_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest120_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest120_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest120_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest120_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest120_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest120_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest120_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest120_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest120_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest120_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest120_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest120_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest120_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest120_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest120_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest120_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest120_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest120_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest120_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest120_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest120_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest120_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest120_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest120_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest120_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest120_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest120_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest120_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest120_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest120_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest120_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest120_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest120_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest120_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest120_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest120_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest120_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest120_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest120_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest120_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest120_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest120_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest120_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest120_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest120_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest120_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest120_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest120_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest120_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest120_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_120
  GROUP BY
    card_id
)
,
his_latest_150 as (
  SELECT
    card_id as id_l150,
    -- mean
    avg(diff_days_lag1)        As his_latest150_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest150_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest150_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest150_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest150_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest150_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest150_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest150_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest150_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest150_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest150_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest150_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest150_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest150_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest150_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest150_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest150_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest150_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest150_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest150_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest150_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest150_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest150_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest150_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest150_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest150_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest150_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest150_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest150_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest150_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest150_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest150_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest150_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest150_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest150_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest150_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest150_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest150_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest150_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest150_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest150_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest150_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest150_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest150_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest150_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest150_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest150_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest150_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest150_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest150_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest150_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest150_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest150_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest150_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest150_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest150_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest150_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest150_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest150_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest150_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_150
  GROUP BY
    card_id
)
,
his_latest_180 as (
  SELECT
    card_id as id_l180,
    -- mean
    avg(diff_days_lag1)        As his_latest180_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest180_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest180_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest180_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest180_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest180_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest180_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest180_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest180_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest180_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest180_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest180_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest180_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest180_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest180_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest180_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest180_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest180_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest180_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest180_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest180_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest180_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest180_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest180_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest180_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest180_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest180_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest180_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest180_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest180_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest180_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest180_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest180_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest180_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest180_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest180_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest180_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest180_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest180_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest180_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest180_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest180_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest180_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest180_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest180_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest180_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest180_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest180_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest180_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest180_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest180_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest180_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest180_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest180_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest180_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest180_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest180_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest180_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest180_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest180_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_180
  GROUP BY
    card_id
)
,
his_latest_360 as (
  SELECT
    card_id as id_l360,
    -- mean
    avg(diff_days_lag1)        As his_latest360_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest360_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest360_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest360_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest360_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest360_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest360_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest360_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest360_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest360_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest360_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest360_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest360_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest360_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest360_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest360_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest360_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest360_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest360_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest360_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest360_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest360_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest360_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest360_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest360_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest360_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest360_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest360_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest360_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest360_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest360_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest360_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest360_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest360_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest360_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest360_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest360_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest360_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest360_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest360_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest360_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest360_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest360_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest360_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest360_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest360_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest360_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest360_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest360_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest360_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest360_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest360_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest360_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest360_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest360_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest360_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest360_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest360_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest360_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest360_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_360
  GROUP BY
    card_id
)
,
his_latest_720 as (
  SELECT
    card_id as id_l720,
    -- mean
    avg(diff_days_lag1)        As his_latest720_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_latest720_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_latest720_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_latest720_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_latest720_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_latest720_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_latest720_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_latest720_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_latest720_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_latest720_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_latest720_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_latest720_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_latest720_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_latest720_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_latest720_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_latest720_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_latest720_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_latest720_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_latest720_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_latest720_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_latest720_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_latest720_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_latest720_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_latest720_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_latest720_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_latest720_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_latest720_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_latest720_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_latest720_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_latest720_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_latest720_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_latest720_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_latest720_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_latest720_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_latest720_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_latest720_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_latest720_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_latest720_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_latest720_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_latest720_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_latest720_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_latest720_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_latest720_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_latest720_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_latest720_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_latest720_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_latest720_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_latest720_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_latest720_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_latest720_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_latest720_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_latest720_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_latest720_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_latest720_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_latest720_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_latest720_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_latest720_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_latest720_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_latest720_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_latest720_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date >= his_latest_720
  GROUP BY
    card_id
)
,
his_first_30 as (
  SELECT
    card_id as id_f30,
    -- mean
    avg(diff_days_lag1)        As his_first30_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_first30_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_first30_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_first30_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_first30_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_first30_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_first30_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_first30_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_first30_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_first30_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_first30_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_first30_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_first30_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_first30_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_first30_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_first30_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_first30_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_first30_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_first30_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_first30_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_first30_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_first30_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_first30_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_first30_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_first30_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_first30_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_first30_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_first30_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_first30_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_first30_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_first30_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_first30_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_first30_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_first30_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_first30_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_first30_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_first30_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_first30_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_first30_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_first30_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_first30_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_first30_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_first30_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_first30_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_first30_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_first30_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_first30_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_first30_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_first30_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_first30_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_first30_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_first30_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_first30_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_first30_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_first30_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_first30_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_first30_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_first30_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_first30_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_first30_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date <= his_first_30
  GROUP BY
    card_id
)
,
his_first_60 as (
  SELECT
    card_id as id_f60,
    -- mean
    avg(diff_days_lag1)        As his_first60_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_first60_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_first60_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_first60_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_first60_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_first60_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_first60_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_first60_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_first60_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_first60_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_first60_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_first60_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_first60_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_first60_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_first60_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_first60_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_first60_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_first60_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_first60_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_first60_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_first60_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_first60_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_first60_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_first60_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_first60_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_first60_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_first60_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_first60_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_first60_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_first60_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_first60_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_first60_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_first60_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_first60_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_first60_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_first60_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_first60_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_first60_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_first60_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_first60_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_first60_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_first60_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_first60_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_first60_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_first60_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_first60_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_first60_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_first60_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_first60_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_first60_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_first60_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_first60_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_first60_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_first60_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_first60_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_first60_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_first60_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_first60_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_first60_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_first60_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date <= his_first_60
  GROUP BY
    card_id
)
,
his_first_90 as (
  SELECT
    card_id as id_f90,
    -- mean
    avg(diff_days_lag1)        As his_first90_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_first90_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_first90_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_first90_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_first90_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_first90_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_first90_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_first90_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_first90_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_first90_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_first90_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_first90_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_first90_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_first90_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_first90_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_first90_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_first90_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_first90_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_first90_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_first90_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_first90_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_first90_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_first90_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_first90_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_first90_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_first90_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_first90_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_first90_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_first90_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_first90_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_first90_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_first90_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_first90_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_first90_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_first90_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_first90_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_first90_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_first90_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_first90_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_first90_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_first90_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_first90_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_first90_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_first90_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_first90_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_first90_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_first90_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_first90_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_first90_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_first90_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_first90_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_first90_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_first90_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_first90_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_first90_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_first90_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_first90_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_first90_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_first90_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_first90_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date <= his_first_90
  GROUP BY
    card_id
)
,
his_first_120 as (
  SELECT
    card_id as id_f120,
    -- mean
    avg(diff_days_lag1)        As his_first120_diff_days_lag1_avg__card_id,
    avg(diff_days_lag2)        As his_first120_diff_days_lag2_avg__card_id,
    avg(diff_days_lag3)        As his_first120_diff_days_lag3_avg__card_id,
    avg(diff_days_lead1)       As his_first120_diff_days_lead1_avg__card_id,
    avg(diff_days_lead2)       As his_first120_diff_days_lead2_avg__card_id,
    avg(diff_days_lead3)       As his_first120_diff_days_lead3_avg__card_id,
    avg(ratio_days_lag1_2)     AS his_first120_ratio_days_lag1_2_avg__card_id,
    avg(ratio_days_lag1_3)     AS his_first120_ratio_days_lag1_3_avg__card_id,
    avg(ratio_days_lag2_3)     AS his_first120_ratio_days_lag2_3_avg__card_id,
    avg(ratio_days_lag1_lead1) AS his_first120_ratio_days_lag1_lead1_avg__card_id,
    avg(ratio_days_lag2_lead2) AS his_first120_ratio_days_lag2_lead2_avg__card_id,
    avg(ratio_days_lag3_lead3) AS his_first120_ratio_days_lag3_lead3_avg__card_id,
    -- max
    max(diff_days_lag1)         As his_first120_diff_days_lag1_max__card_id,
    max(diff_days_lag2)         As his_first120_diff_days_lag2_max__card_id,
    max(diff_days_lag3)         As his_first120_diff_days_lag3_max__card_id,
    max(diff_days_lead1)        As his_first120_diff_days_lead1_max__card_id,
    max(diff_days_lead2)        As his_first120_diff_days_lead2_max__card_id,
    max(diff_days_lead3)        As his_first120_diff_days_lead3_max__card_id,
    max(ratio_days_lag1_2)      AS his_first120_ratio_days_lag1_2_max__card_id,
    max(ratio_days_lag1_3)      AS his_first120_ratio_days_lag1_3_max__card_id,
    max(ratio_days_lag2_3)      AS his_first120_ratio_days_lag2_3_max__card_id,
    max(ratio_days_lag1_lead1)  AS his_first120_ratio_days_lag1_lead1_max__card_id,
    max(ratio_days_lag2_lead2)  AS his_first120_ratio_days_lag2_lead2_max__card_id,
    max(ratio_days_lag3_lead3)  AS his_first120_ratio_days_lag3_lead3_max__card_id,
    -- min
    min(diff_days_lag1)        As his_first120_diff_days_lag1_min__card_id,
    min(diff_days_lag2)        As his_first120_diff_days_lag2_min__card_id,
    min(diff_days_lag3)        As his_first120_diff_days_lag3_min__card_id,
    min(diff_days_lead1)       As his_first120_diff_days_lead1_min__card_id,
    min(diff_days_lead2)       As his_first120_diff_days_lead2_min__card_id,
    min(diff_days_lead3)       As his_first120_diff_days_lead3_min__card_id,
    min(ratio_days_lag1_2)     AS his_first120_ratio_days_lag1_2_min__card_id,
    min(ratio_days_lag1_3)     AS his_first120_ratio_days_lag1_3_min__card_id,
    min(ratio_days_lag2_3)     AS his_first120_ratio_days_lag2_3_min__card_id,
    min(ratio_days_lag1_lead1) AS his_first120_ratio_days_lag1_lead1_min__card_id,
    min(ratio_days_lag2_lead2) AS his_first120_ratio_days_lag2_lead2_min__card_id,
    min(ratio_days_lag3_lead3) AS his_first120_ratio_days_lag3_lead3_min__card_id,

    -- ptp
    max(diff_days_lag1) - min(diff_days_lag1)               As his_first120_diff_days_lag1_ptp__card_id,
    max(diff_days_lag2) - min(diff_days_lag2)               As his_first120_diff_days_lag2_ptp__card_id,
    max(diff_days_lag3) - min(diff_days_lag3)               As his_first120_diff_days_lag3_ptp__card_id,
    max(diff_days_lead1) - min(diff_days_lead1)             As his_first120_diff_days_lead1_ptp__card_id,
    max(diff_days_lead2) - min(diff_days_lead2)             As his_first120_diff_days_lead2_ptp__card_id,
    max(diff_days_lead3) - min(diff_days_lead3)             As his_first120_diff_days_lead3_ptp__card_id,
    max(ratio_days_lag1_2) - min(ratio_days_lag1_2)         AS his_first120_ratio_days_lag1_2_ptp__card_id,
    max(ratio_days_lag1_3) - min(ratio_days_lag1_3)         AS his_first120_ratio_days_lag1_3_ptp__card_id,
    max(ratio_days_lag2_3) - min(ratio_days_lag2_3)         AS his_first120_ratio_days_lag2_3_ptp__card_id,
    max(ratio_days_lag1_lead1) - min(ratio_days_lag1_lead1) AS his_first120_ratio_days_lag1_lead1_ptp__card_id,
    max(ratio_days_lag2_lead2) - min(ratio_days_lag2_lead2) AS his_first120_ratio_days_lag2_lead2_ptp__card_id,
    max(ratio_days_lag3_lead3) - min(ratio_days_lag3_lead3) AS his_first120_ratio_days_lag3_lead3_ptp__card_id,

    -- std
    STDDEV_SAMP(diff_days_lag1)        As his_first120_diff_days_lag1_std__card_id,
    STDDEV_SAMP(diff_days_lag2)        As his_first120_diff_days_lag2_std__card_id,
    STDDEV_SAMP(diff_days_lag3)        As his_first120_diff_days_lag3_std__card_id,
    STDDEV_SAMP(diff_days_lead1)       As his_first120_diff_days_lead1_std__card_id,
    STDDEV_SAMP(diff_days_lead2)       As his_first120_diff_days_lead2_std__card_id,
    STDDEV_SAMP(diff_days_lead3)       As his_first120_diff_days_lead3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_2)     AS his_first120_ratio_days_lag1_2_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_3)     AS his_first120_ratio_days_lag1_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_3)     AS his_first120_ratio_days_lag2_3_std__card_id,
    STDDEV_SAMP(ratio_days_lag1_lead1) AS his_first120_ratio_days_lag1_lead1_std__card_id,
    STDDEV_SAMP(ratio_days_lag2_lead2) AS his_first120_ratio_days_lag2_lead2_std__card_id,
    STDDEV_SAMP(ratio_days_lag3_lead3) AS his_first120_ratio_days_lag3_lead3_std__card_id
  from hori.auth_0
  where
    p_date <= his_first_120
  GROUP BY
    card_id
)
,
result as (
  SELECT
    *
  FROM his_latest_30 as t1
  INNER JOIN his_latest_60 as t2
    ON t1.card_id = t2.id_l60
  INNER JOIN his_latest_90 as t3
    ON t1.card_id = t3.id_l90
  INNER JOIN his_latest_120 as t4
    ON t1.card_id = t4.id_l120
  INNER JOIN his_latest_150 as t5
    ON t1.card_id = t5.id_l150
  INNER JOIN his_latest_180 as t6
    ON t1.card_id = t6.id_l180
  INNER JOIN his_latest_360 as t7
    ON t1.card_id = t7.id_l360
  INNER JOIN his_latest_720 as t8
    ON t1.card_id = t8.id_l720
  INNER JOIN his_first_30 as t9
    ON t1.card_id = t9.id_f30
  INNER JOIN his_first_60 as t10
    ON t1.card_id = t10.id_f60
  INNER JOIN his_first_90 as t11
    ON t1.card_id = t11.id_f90
  INNER JOIN his_first_120 as t12
    ON t1.card_id = t12.id_f120
)
SELECT
*
FROM result
;
