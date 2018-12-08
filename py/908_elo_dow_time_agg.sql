-- feat_no 908
-- #========================================================================
-- # DOw_timezone aggregation
-- #========================================================================

# month agg
WITH
  latest_year_agg as (
  SELECT
    card_id,
    dow,
    timezone,
    count(1) as transactions_cnt__card_id_latest_year_dow_timezone,
    avg(purchase_amount) as purchase_amount_mean__card_id_latest_year_dow_timezone,
    avg(installments) as installments_mean__card_id_latest_year_dow_timezone,

    count(distinct cast(purchase_date as date)) as date_nuq__card_id_latest_year_dow_timezone,
    count(distinct timezone) as timezone_nuq__card_id_latest_year_dow_timezone,
    count(distinct month) as month_nuq__card_id_latest_year_dow_timezone,
    count(distinct merchant_category_id) as merchant_category_id_nuq__card_id_latest_year_dow_timezone,
    count(distinct merchant_id) as merchant_id_nuq__card_id_latest_year_dow_timezone

  FROM
    hori.historical_time_detail
  where
    authorized_flag is True
    -- authorized_flag is false
    and latest_month_no<=12 -- latest 1year
  GROUP BY
    card_id,
    dow,
    timezone
)
,
  month_agg as (
  SELECT
    card_id,
    latest_month_no,
    dow,
    timezone,
    count(1) as transactions_cnt__card_id_month_dow_timezone,
    sum(purchase_amount) as purchase_amount_sum__card_id_month_dow_timezone,
    avg(purchase_amount) as purchase_amount_mean__card_id_month_dow_timezone,
    max(purchase_amount) as purchase_amount_max__card_id_month_dow_timezone,
    min(purchase_amount) as purchase_amount_min__card_id_month_dow_timezone,
    STDDEV_SAMP(purchase_amount) as purchase_amount_std__card_id_month_dow_timezone,

    sum(installments) as installments_sum__card_id_month_dow_timezone,
    avg(installments) as installments_mean__card_id_month_dow_timezone,
    max(installments) as installments_max__card_id_month_dow_timezone,
    min(installments) as installments_min__card_id_month_dow_timezone,
    STDDEV_SAMP(installments) as installments_std__card_id_month_dow_timezone,
    count(distinct installments) as installments_nuq__card_id_month_dow_timezone,

    count(distinct cast(purchase_date as date)) as date_nuq__card_id_month_dow_timezone,
    count(distinct timezone) as timezone_nuq__card_id_month_dow_timezone,
    count(distinct merchant_category_id) as merchant_category_id_nuq__card_id_month_dow_timezone,
    count(distinct merchant_id) as merchant_id_nuq__card_id_month_dow_timezone

  FROM
    hori.historical_time_detail
  where
    authorized_flag is True
    -- authorized_flag is false
    and latest_month_no<=2 -- latest 3month
  GROUP BY
    card_id,
    latest_month_no,
    dow,
    timezone
)
,
month_agg_ratio as (
  SELECT
    t1.card_id,
    latest_month_no,
    t1.dow,
    t1.timezone,
    transactions_cnt__card_id_month_dow_timezone,
    purchase_amount_sum__card_id_month_dow_timezone,
    purchase_amount_mean__card_id_month_dow_timezone,
    purchase_amount_max__card_id_month_dow_timezone,
    purchase_amount_min__card_id_month_dow_timezone,
    purchase_amount_std__card_id_month_dow_timezone,

    installments_sum__card_id_month_dow_timezone,
    installments_mean__card_id_month_dow_timezone,
    installments_max__card_id_month_dow_timezone,
    installments_min__card_id_month_dow_timezone,
    installments_std__card_id_month_dow_timezone,
    installments_nuq__card_id_month_dow_timezone,

    date_nuq__card_id_month_dow_timezone,
    timezone_nuq__card_id_month_dow_timezone,
    merchant_category_id_nuq__card_id_month_dow_timezone,
    merchant_id_nuq__card_id_month_dow_timezone,

    # year
    transactions_cnt__card_id_latest_year_dow_timezone,
    purchase_amount_mean__card_id_latest_year_dow_timezone,
    installments_mean__card_id_latest_year_dow_timezone,

    date_nuq__card_id_latest_year_dow_timezone,
    timezone_nuq__card_id_latest_year_dow_timezone,
    month_nuq__card_id_latest_year_dow_timezone,
    merchant_category_id_nuq__card_id_latest_year_dow_timezone,
    merchant_id_nuq__card_id_latest_year_dow_timezone,

    # ratio
    transactions_cnt__card_id_month_dow_timezone / (transactions_cnt__card_id_latest_year_dow_timezone+0.000001) as transactions_cnt__card_id_month_ratio_year_dow_timezone,
    purchase_amount_mean__card_id_month_dow_timezone / (purchase_amount_mean__card_id_latest_year_dow_timezone+0.000001) as purchase_amount_mean__card_id_month_ratio_year_dow_timezone,
    installments_mean__card_id_month_dow_timezone / (installments_mean__card_id_latest_year_dow_timezone+0.000001) as installments_mean__card_id_month_ratio_year_dow_timezone,

    date_nuq__card_id_month_dow_timezone / (date_nuq__card_id_latest_year_dow_timezone+0.000001)  as  date_nuq__card_id_month_ratio_year_dow_timezone ,
    timezone_nuq__card_id_month_dow_timezone  / (timezone_nuq__card_id_latest_year_dow_timezone+0.000001)   as timezone_nuq__card_id_month_ratio_year_dow_timezone ,
    merchant_category_id_nuq__card_id_month_dow_timezone / (merchant_category_id_nuq__card_id_latest_year_dow_timezone+0.000001)   as merchant_category_id_nuq__card_id_month_ratio_year_dow_timezone,
    merchant_id_nuq__card_id_month_dow_timezone / (merchant_id_nuq__card_id_latest_year_dow_timezone+0.000001)  as merchant_id_nuq__card_id_month_ratio_year_dow_timezone

  FROM month_agg as t1
  INNER JOIN latest_year_agg as t2
    ON t1.card_id = t2.card_id and t1.dow = t2.dow and t1.timezone = t2.timezone
)

SELECT
  *
FROM month_agg_ratio
;
