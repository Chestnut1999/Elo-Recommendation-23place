-- feat_no 907
-- #========================================================================
-- # Dow aggregation
-- #========================================================================
# month agg

WITH
  latest_year_agg as (
  SELECT
    card_id,
    dow,
    count(1) as transactions_cnt__card_id_latest_year_dow,
    avg(purchase_amount) as purchase_amount_mean__card_id_latest_year_dow,
    avg(installments) as installments_mean__card_id_latest_year_dow,

    count(distinct cast(purchase_date as date)) as date_nuq__card_id_latest_year_dow,
    count(distinct timezone) as timezone_nuq__card_id_latest_year_dow,
    count(distinct month) as month_nuq__card_id_latest_year_dow,
    count(distinct merchant_category_id) as merchant_category_id_nuq__card_id_latest_year_dow,
    count(distinct merchant_id) as merchant_id_nuq__card_id_latest_year_dow

  FROM
    hori.historical_time_detail
  where
    authorized_flag is True
    -- authorized_flag is false
    and latest_month_no<=12 -- latest 1year
  GROUP BY
    card_id,
    dow
)
,
  month_agg as (
  SELECT
    card_id,
    latest_month_no,
    dow,
    count(1) as transactions_cnt__card_id_month_dow,
    sum(purchase_amount) as purchase_amount_sum__card_id_month_dow,
    avg(purchase_amount) as purchase_amount_mean__card_id_month_dow,
    max(purchase_amount) as purchase_amount_max__card_id_month_dow,
    min(purchase_amount) as purchase_amount_min__card_id_month_dow,
    STDDEV_SAMP(purchase_amount) as purchase_amount_std__card_id_month_dow,

    sum(installments) as installments_sum__card_id_month_dow,
    avg(installments) as installments_mean__card_id_month_dow,
    max(installments) as installments_max__card_id_month_dow,
    min(installments) as installments_min__card_id_month_dow,
    STDDEV_SAMP(installments) as installments_std__card_id_month_dow,
    count(distinct installments) as installments_nuq__card_id_month_dow,

    count(distinct cast(purchase_date as date)) as date_nuq__card_id_month_dow,
    count(distinct timezone) as timezone_nuq__card_id_month_dow,
    count(distinct merchant_category_id) as merchant_category_id_nuq__card_id_month_dow,
    count(distinct merchant_id) as merchant_id_nuq__card_id_month_dow

  FROM
    hori.historical_time_detail
  where
    authorized_flag is True
    -- authorized_flag is false
    and latest_month_no<=12 -- latest 1year
  GROUP BY
    card_id,
    latest_month_no,
    dow
)
,
month_agg_ratio as (
  SELECT
    t1.card_id,
    latest_month_no,
    t1.dow,
    transactions_cnt__card_id_month_dow,
    purchase_amount_sum__card_id_month_dow,
    purchase_amount_mean__card_id_month_dow,
    purchase_amount_max__card_id_month_dow,
    purchase_amount_min__card_id_month_dow,
    purchase_amount_std__card_id_month_dow,

    installments_sum__card_id_month_dow,
    installments_mean__card_id_month_dow,
    installments_max__card_id_month_dow,
    installments_min__card_id_month_dow,
    installments_std__card_id_month_dow,
    installments_nuq__card_id_month_dow,

    date_nuq__card_id_month_dow,
    timezone_nuq__card_id_month_dow,
    merchant_category_id_nuq__card_id_month_dow,
    merchant_id_nuq__card_id_month_dow,

    # year
    transactions_cnt__card_id_latest_year_dow,
    purchase_amount_mean__card_id_latest_year_dow,
    installments_mean__card_id_latest_year_dow,

    date_nuq__card_id_latest_year_dow,
    timezone_nuq__card_id_latest_year_dow,
    month_nuq__card_id_latest_year_dow,
    merchant_category_id_nuq__card_id_latest_year_dow,
    merchant_id_nuq__card_id_latest_year_dow,

    # ratio
    transactions_cnt__card_id_month_dow / (transactions_cnt__card_id_latest_year_dow+0.000001) as transactions_cnt__card_id_month_ratio_year_dow,
    purchase_amount_mean__card_id_month_dow / (purchase_amount_mean__card_id_latest_year_dow+0.000001) as purchase_amount_mean__card_id_month_ratio_year_dow,
    installments_mean__card_id_month_dow / (installments_mean__card_id_latest_year_dow+0.000001) as installments_mean__card_id_month_ratio_year_dow,

    date_nuq__card_id_month_dow / (date_nuq__card_id_latest_year_dow+0.000001)  as  date_nuq__card_id_month_ratio_year_dow ,
    timezone_nuq__card_id_month_dow  / (timezone_nuq__card_id_latest_year_dow+0.000001)   as timezone_nuq__card_id_month_ratio_year_dow ,
    merchant_category_id_nuq__card_id_month_dow / (merchant_category_id_nuq__card_id_latest_year_dow+0.000001)   as merchant_category_id_nuq__card_id_month_ratio_year_dow,
    merchant_id_nuq__card_id_month_dow / (merchant_id_nuq__card_id_latest_year_dow+0.000001)  as merchant_id_nuq__card_id_month_ratio_year_dow

  FROM month_agg as t1
  INNER JOIN latest_year_agg as t2
    ON t1.card_id = t2.card_id and t1.dow = t2.dow
)


SELECT
  *
FROM month_agg_ratio
;
