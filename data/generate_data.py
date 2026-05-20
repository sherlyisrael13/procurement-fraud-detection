import pandas as pd
import numpy as np
from faker import Faker
import sqlite3
import random
from datetime import datetime, timedelta
import os

# Project  : Procurement Fraud Detection System with Explainable AI
# Author   : Josephine Sherly P

fake = Faker()
random.seed(42)
np.random.seed(42)

DEPARTMENTS = ['IT', 'Finance', 'HR', 'Operations', 'Marketing', 'Legal', 'Procurement']

CATEGORIES = {
    'Software':        (500,   15000),
    'Hardware':        (1000,  50000),
    'Consulting':      (2000,  80000),
    'Office Supplies': (50,    2000),
    'Travel':          (200,   8000),
    'Maintenance':     (300,   10000),
    'Training':        (500,   5000),
}

VENDORS   = [fake.company() for _ in range(80)]
APPROVERS = [fake.name()    for _ in range(15)]

DEPT_RISK = {
    'IT': 0.3, 'Finance': 0.5, 'HR': 0.1,
    'Operations': 0.2, 'Marketing': 0.3,
    'Legal': 0.4, 'Procurement': 0.6
}

def generate_normal_order(order_id):
    category   = random.choice(list(CATEGORIES.keys()))
    low, high  = CATEGORIES[category]
    amount     = round(random.uniform(low, high), 2)
    created    = fake.date_time_between(start_date='-1y', end_date='now')
    lag        = random.randint(4, 72)
    approved   = created + timedelta(hours=lag)
    department = random.choice(DEPARTMENTS)

    return {
        'order_id':              f'PO-{order_id:06d}',
        'vendor':                random.choice(VENDORS),
        'department':            department,
        'category':              category,
        'amount':                amount,
        'created_at':            created.strftime('%Y-%m-%d %H:%M:%S'),
        'approved_at':           approved.strftime('%Y-%m-%d %H:%M:%S'),
        'approver':              random.choice(APPROVERS),
        'approval_lag_hours':    lag,
        'department_risk_score': DEPT_RISK[department],
        'is_anomaly':            0,
        'anomaly_reason':        'none'
    }

def inject_anomaly(order, order_id):
    anomaly_type = random.choice([
        'inflated_amount',
        'unknown_vendor',
        'weekend_approval',
        'duplicate_pattern',
        'unusual_lag'
    ])
    order['is_anomaly']     = 1
    order['anomaly_reason'] = anomaly_type

    if anomaly_type == 'inflated_amount':
        _, high = CATEGORIES[order['category']]
        order['amount'] = round(high * random.uniform(4.0, 10.0), 2)

    elif anomaly_type == 'unknown_vendor':
        order['vendor'] = f'UNKNOWN_VENDOR_{order_id}'

    elif anomaly_type == 'weekend_approval':
        base = fake.date_time_between(start_date='-1y', end_date='now')
        while base.weekday() not in [5, 6]:
            base = fake.date_time_between(start_date='-1y', end_date='now')
        order['approved_at']        = base.strftime('%Y-%m-%d %H:%M:%S')
        order['approval_lag_hours'] = random.randint(0, 2)

    elif anomaly_type == 'duplicate_pattern':
        order['amount']             = 9999.99
        order['category']           = 'Consulting'
        order['approval_lag_hours'] = random.randint(0, 1)

    elif anomaly_type == 'unusual_lag':
        order['approval_lag_hours'] = random.randint(200, 720)

    return order

def generate_dataset(n=10000, anomaly_rate=0.05):
    print(f"Generating {n} purchase orders...")
    records = []

    for i in range(1, n + 1):
        order = generate_normal_order(i)
        if random.random() < anomaly_rate:
            order = inject_anomaly(order, i)
        records.append(order)

    df = pd.DataFrame(records)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/procurement_data.csv', index=False)

    conn = sqlite3.connect('data/procurement.db')
    df.to_sql('orders', conn, if_exists='replace', index=False)
    conn.close()

    total = int(df['is_anomaly'].sum())
    print(f"\n✅ Done.")
    print(f"   Total orders  : {n}")
    print(f"   Anomalies     : {total} ({100*total/n:.1f}%)")
    print(f"   CSV saved     : data/procurement_data.csv")
    print(f"   DB saved      : data/procurement.db")
    print(f"\nAnomaly breakdown:")
    print(df[df['is_anomaly']==1]['anomaly_reason'].value_counts().to_string())
    return df

if __name__ == '__main__':
    df = generate_dataset(n=10000, anomaly_rate=0.05)
    sample = df[df['is_anomaly']==1].iloc[0]
    print(f"\nSample flagged order:")
    print(f"  Order ID : {sample['order_id']}")
    print(f"  Vendor   : {sample['vendor']}")
    print(f"  Amount   : {sample['amount']:,.2f}")
    print(f"  Reason   : {sample['anomaly_reason']}")