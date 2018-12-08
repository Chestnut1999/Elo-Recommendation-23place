-- feat_no 906
-- #========================================================================
-- # Month aggregation
-- #========================================================================
# month agg

WITH
  latest_year_agg as (
  SELECT
    card_id,
    count(1) as transactions_cnt__card_id_latest_year,
    avg(purchase_amount) as purchase_amount_mean__card_id_latest_year,
    avg(installments) as installments_mean__card_id_latest_year,

    count(distinct cast(purchase_date as date)) as date_nuq__card_id_latest_year,
    count(distinct dow) as dow_nuq__card_id_latest_year,
    count(distinct timezone) as timezone_nuq__card_id_latest_year,
    count(distinct dow_timezone) as dow_timezone_nuq__card_id_latest_year,
    count(distinct month) as month_nuq__card_id_latest_year,
    count(distinct merchant_category_id) as merchant_category_id_nuq__card_id_latest_year,
    count(distinct merchant_id) as merchant_id_nuq__card_id_latest_year

  FROM
    hori.historical_time_detail
  where
    authorized_flag is True
    -- authorized_flag is false
    and latest_month_no<=12 -- latest 1year
  GROUP BY
    card_id
)
,
  month_agg as (
  SELECT
    card_id,
    latest_month_no,
    count(1) as transactions_cnt__card_id_month,
    sum(purchase_amount) as purchase_amount_sum__card_id_month,
    avg(purchase_amount) as purchase_amount_mean__card_id_month,
    max(purchase_amount) as purchase_amount_max__card_id_month,
    min(purchase_amount) as purchase_amount_min__card_id_month,
    STDDEV_SAMP(purchase_amount) as purchase_amount_std__card_id_month,

    sum(installments) as installments_sum__card_id_month,
    avg(installments) as installments_mean__card_id_month,
    max(installments) as installments_max__card_id_month,
    min(installments) as installments_min__card_id_month,
    STDDEV_SAMP(installments) as installments_std__card_id_month,
    count(distinct installments) as installments_nuq__card_id_month,

    count(distinct cast(purchase_date as date)) as date_nuq__card_id_month,
    count(distinct dow) as dow_nuq__card_id_month,
    count(distinct timezone) as timezone_nuq__card_id_month,
    count(distinct dow_timezone) as dow_timezone_nuq__card_id_month,
    count(distinct merchant_category_id) as merchant_category_id_nuq__card_id_month,
    count(distinct merchant_id) as merchant_id_nuq__card_id_month

  FROM
    hori.historical_time_detail
  where
    authorized_flag is True
    -- authorized_flag is false
    and latest_month_no<=12 -- latest 1year
  GROUP BY
    card_id,
    latest_month_no
)
,
month_agg_ratio as (
  SELECT
    t1.card_id,
    latest_month_no,
    transactions_cnt__card_id_month,
    purchase_amount_sum__card_id_month,
    purchase_amount_mean__card_id_month,
    purchase_amount_max__card_id_month,
    purchase_amount_min__card_id_month,
    purchase_amount_std__card_id_month,

    installments_sum__card_id_month,
    installments_mean__card_id_month,
    installments_max__card_id_month,
    installments_min__card_id_month,
    installments_std__card_id_month,
    installments_nuq__card_id_month,

    date_nuq__card_id_month,
    dow_nuq__card_id_month,
    timezone_nuq__card_id_month,
    dow_timezone_nuq__card_id_month,
    merchant_category_id_nuq__card_id_month,
    merchant_id_nuq__card_id_month,

    # year
    transactions_cnt__card_id_latest_year,
    purchase_amount_mean__card_id_latest_year,
    installments_mean__card_id_latest_year,

    date_nuq__card_id_latest_year,
    dow_nuq__card_id_latest_year,
    timezone_nuq__card_id_latest_year,
    dow_timezone_nuq__card_id_latest_year,
    month_nuq__card_id_latest_year,
    merchant_category_id_nuq__card_id_latest_year,
    merchant_id_nuq__card_id_latest_year,

    # ratio
    transactions_cnt__card_id_month / (transactions_cnt__card_id_latest_year+0.000001) as transactions_cnt__card_id_month_ratio_year,
    purchase_amount_mean__card_id_month / (purchase_amount_mean__card_id_latest_year+0.000001) as purchase_amount_mean__card_id_month_ratio_year,
    installments_mean__card_id_month / (installments_mean__card_id_latest_year+0.000001) as installments_mean__card_id_month_ratio_year,

    date_nuq__card_id_month / (date_nuq__card_id_latest_year+0.000001)  as  date_nuq__card_id_month_ratio_year ,
    dow_nuq__card_id_month / (dow_nuq__card_id_latest_year+0.000001)   as  dow_nuq__card_id_month_ratio_year,
    timezone_nuq__card_id_month  / (timezone_nuq__card_id_latest_year+0.000001)   as timezone_nuq__card_id_month_ratio_year ,
    dow_timezone_nuq__card_id_month / (dow_timezone_nuq__card_id_latest_year+0.000001)  as dow_timezone_nuq__card_id_month_ratio_year  ,
    merchant_category_id_nuq__card_id_month / (merchant_category_id_nuq__card_id_latest_year+0.000001)   as merchant_category_id_nuq__card_id_month_ratio_year,
    merchant_id_nuq__card_id_month / (merchant_id_nuq__card_id_latest_year+0.000001)  as merchant_id_nuq__card_id_month_ratio_year

  FROM month_agg as t1
  INNER JOIN latest_year_agg as t2
    ON t1.card_id = t2.card_id
)


SELECT
  *
FROM month_agg_ratio
;
