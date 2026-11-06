"""Assert core Part A checkpoints and print one exact success line."""
from pathlib import Path
import sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
customers=pd.read_csv(ROOT/'data/clean/customers.csv')
products=pd.read_csv(ROOT/'data/clean/products.csv')
orders=pd.read_csv(ROOT/'data/clean/orders.csv')
assert len(customers)==1200
assert customers.city.nunique()==6
assert customers.age.notna().all()
assert len(products)==40
assert len(orders)==12309 and orders.order_id.nunique()==12309
assert orders.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199}
assert orders.loc[orders.status=='completed','delivery_minutes'].notna().all()
assert orders.loc[orders.status=='cancelled','delivery_minutes'].isna().all()
assert round(orders.loc[orders.status=='completed','order_amount'].sum(),2)==2998668.80
with sqlite3.connect(ROOT/'data/instamart.db') as con:
    names={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'customers','products','orders'} <= names
print('ALL CHECKPOINTS PASSED')
