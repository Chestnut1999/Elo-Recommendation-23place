-- TODO
-- feat_no 901
-- new latestとhistorical latestのdiff
-- new latestとhistorical most pastのdiff
-- new most pastとhistorical latestのdiff
-- new most pastとhistorical most pastのdiff
with latest as(
select
card_id,
max(purchase_date) as latest
from hori.new
)
,
hist_last as(
select 
card_id,
max(purchase_date) as hist_last
from hori.elo_historical
)
select
distinct
date_diff(cast(latest as date), cast(hist_last as date), DAY) as diff_new_hist
from latest as t1
inner join hist_last as t2
on t1.card_id = t2.card_id
order by diff_new_hist
;
