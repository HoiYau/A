
import pandas as pd
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestClassifier
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def forecast_sales():
    conn = get_db_connection()
    sales_df = pd.read_sql_query("SELECT * FROM sales", conn)
    conn.close()

    sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])
    sales_df.set_index('sale_date', inplace=True)
    
    daily_sales = sales_df['quantity'].resample('D').sum()

    model = SARIMAX(daily_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit()
    
    forecast = results.get_forecast(steps=30)
    return forecast.predicted_mean

def predict_purchase(user_id, product_ids):
    conn = get_db_connection()
    # This is a simplified example. In a real-world scenario, you'd have a much more complex feature engineering pipeline.
    # For now, we'll just create a dummy classifier.
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    products_df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    
    # Dummy features and target
    X = pd.DataFrame({
        'user_id': users_df['id'].repeat(len(products_df)),
        'product_id': list(products_df['id']) * len(users_df)
    })
    y = (X['user_id'] + X['product_id']) % 2 # Dummy target

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    new_data = pd.DataFrame({
        'user_id': [user_id] * len(product_ids),
        'product_id': product_ids
    })
    
    predictions = model.predict_proba(new_data)
    return dict(zip(product_ids, predictions[:, 1]))

def recommend_products(user_id):
    conn = get_db_connection()
    sales_df = pd.read_sql_query("SELECT user_id, product_id, quantity FROM sales", conn)
    conn.close()

    reader = Reader(rating_scale=(1, sales_df['quantity'].max()))
    data = Dataset.load_from_df(sales_df[['user_id', 'product_id', 'quantity']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.25)
    
    model = SVD()
    model.fit(trainset)
    
    product_ids = sales_df['product_id'].unique()
    
    predictions = []
    for product_id in product_ids:
        predictions.append(model.predict(user_id, product_id))
        
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    return [rec.iid for rec in recommendations[:10]]
