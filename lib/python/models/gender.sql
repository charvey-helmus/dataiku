select "gender" , count(*) as cnt
from
DATAIKU.PUBLIC.CUSTOMERLIFETIMEVALUEONSNOWFLAKE_PREDICTION

group by "gender"
