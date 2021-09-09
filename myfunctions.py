import dataiku

def create_target(row):
    revenue = row['revenue']
    v = int(dataiku.get_custom_variables()["revenue_value"])
    if revenue>=v:
        target=1
    elif revenue<v:
        target=0
    else:
        target = revenue
    return target
