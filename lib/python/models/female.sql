select *
from {{ ref('gender') }}
where "gender" = 'F'
