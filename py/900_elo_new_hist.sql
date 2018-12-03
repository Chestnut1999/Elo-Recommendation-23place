with
lag1 as(
select
card_id,
purchase_date,
lag(purchase_date, 1)over(partition by card_id order by purchase_date , merchant_id ) lag_1,
lag(purchase_date, 2)over(partition by card_id order by purchase_date , merchant_id ) lag_2,
lag(purchase_date, 3)over(partition by card_id order by purchase_date , merchant_id ) lag_3
from `hori.elo_historical`
)
select
card_id,
purchase_date ,
lag_1,
date_diff(cast(purchase_date as date), cast(lag_1 as date), day)
from lag1
limit 100
;

--  #========================================================================
--  # diff ratio days feature
--  #========================================================================
with
lag_lead as(
select
card_id,
purchase_date,
lag(purchase_date, 1)over(partition by card_id order by purchase_date , merchant_id ) lag_1,
lag(purchase_date, 2)over(partition by card_id order by purchase_date , merchant_id ) lag_2,
lag(purchase_date, 3)over(partition by card_id order by purchase_date , merchant_id ) lag_3,
lead(purchase_date, 1)over(partition by card_id order by purchase_date , merchant_id ) lead_1,
lead(purchase_date, 2)over(partition by card_id order by purchase_date , merchant_id ) lead_2,
lead(purchase_date, 3)over(partition by card_id order by purchase_date , merchant_id ) lead_3
from `hori.elo_historical` 
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
diff_days_lag1 / diff_days_lag2 as ratio_days_lag1_2,
diff_days_lag1 / diff_days_lag3 as ratio_days_lag1_3,
diff_days_lag2 / diff_days_lag3 as ratio_days_lag2_3,
diff_days_lag1 / diff_days_lead1 as ratio_days_lag1_lead1,
diff_days_lag2 / diff_days_lead2 as ratio_days_lag2_lead2,
diff_days_lag3 / diff_days_lead3 as ratio_days_lag3_lead3
from diff_days
)
-- diff, ratio featureを作成したら、後は集計して粒度を合わせる
select *
from diff_days
;
