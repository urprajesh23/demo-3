from flask import Flask, render_template, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Global variables to store stock data
stocks_data = {}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/<category>")
def category_page(category):
    if category not in stocks_data:
        return redirect(url_for("home"))

    data = stocks_data[category]
    return render_template(
        "category.html",
        title=data["title"],
        stocks=data["stocks"],
        metric=data["metric"]
    )


if __name__ == "__main__":
    # Load and process data
    file_path = 'fnl_stockds_filled.xlsx'
    df = pd.read_excel(file_path)

    if 'Name ' in df.columns:
        df['Name'] = df['Name'].str.replace(' ', '', regex=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    features = ['Sales Growth %', 'Profit Var 5Yrs %', 'Profit Var 3Yrs %', 'ROE %',
                'P/E', 'Ind PE', 'Debt/Eq', 'CMP Rs.', 'Current Ratio', 'Div Yld%',
                'EPS 12M Rs.', 'ROA 12M %', 'CMP/Sales', 'ROCE %']
    features = [f for f in features if f in df.columns]

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # Existing Features (Long-Term, Short-Term, Dividend, Penny Stocks)
    # Long-Term Potential
    y_long_term = df['ROE %'] + df['Sales Growth %']
    rf_long_term = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_long_term.fit(df_scaled, y_long_term)
    df['Long_Term_Potential'] = rf_long_term.predict(df_scaled)
    stocks_data['longterm'] = {
        "title": "Top 5 Long-Term Potential Stocks",
        "stocks": df[['Name', 'Long_Term_Potential', 'CMP Rs.', 'ROE %', 'P/E']]
        .sort_values(by='Long_Term_Potential', ascending=False)
        .head(5)
        .to_dict(orient='records'),
        "metric": "Long_Term_Potential"
    }

    # Short-Term Potential
    y_short_term = df['Sales Growth %'] + df['Profit Var 3Yrs %']
    rf_short_term = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_short_term.fit(df_scaled, y_short_term)
    df['Short_Term_Potential'] = rf_short_term.predict(df_scaled)
    stocks_data['shortterm'] = {
        "title": "Top 5 Short-Term Potential Stocks",
        "stocks": df[['Name', 'Short_Term_Potential', 'CMP Rs.', 'ROE %', 'P/E']]
        .sort_values(by='Short_Term_Potential', ascending=False)
        .head(5)
        .to_dict(orient='records'),
        "metric": "Short_Term_Potential"
    }

    # Dividend Yield Stocks
    y_dividend = df['Div Yld%']
    rf_dividend = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_dividend.fit(df_scaled, y_dividend)
    df['Dividend_Potential'] = rf_dividend.predict(df_scaled)
    stocks_data['dividend'] = {
        "title": "Top 5 Dividend Yield Stocks",
        "stocks": df[['Name', 'Dividend_Potential', 'CMP Rs.', 'ROE %', 'P/E']]
        .sort_values(by='Dividend_Potential', ascending=False)
        .head(5)
        .to_dict(orient='records'),
        "metric": "Dividend_Potential"
    }

    # Penny Stocks
    penny_stock_threshold = 100
    penny_stocks_df = df[df['CMP Rs.'] < penny_stock_threshold]
    y_penny = penny_stocks_df['EPS 12M Rs.']
    penny_scaled = pd.DataFrame(scaler.fit_transform(penny_stocks_df[features]), columns=features)
    rf_penny = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_penny.fit(penny_scaled, y_penny)
    penny_stocks_df['Return_Potential'] = rf_penny.predict(penny_scaled)
    stocks_data['penny'] = {
        "title": "Top 5 Penny Stocks",
        "stocks": penny_stocks_df[['Name', 'Return_Potential', 'CMP Rs.', 'ROE %', 'P/E']]
        .sort_values(by='Return_Potential', ascending=False)
        .head(5)
        .to_dict(orient='records'),
        "metric": "Return_Potential"
    }

    # New Features (Best Performing Stocks, Greater Returns)
    # Best Performing Stocks
    if 'ROCE %' in df.columns:
        y_best_performing = df['ROCE %']
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, y_best_performing, test_size=0.3, random_state=42)
        rf_best_performing = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_best_performing.fit(X_train, y_train)
        df['Performance_Potential'] = rf_best_performing.predict(df_scaled)
        stocks_data['bestperforming'] = {
            "title": "Top 5 Best Performing Stocks",
            "stocks": df[['Name', 'Performance_Potential', 'CMP Rs.', 'ROE %', 'P/E']]
            .sort_values(by='Performance_Potential', ascending=False)
            .head(5)
            .to_dict(orient='records'),
            "metric": "Performance_Potential"
        }

    # Greater Returns
    y_greater_returns = df['ROCE %'] + df['EPS 12M Rs.']
    rf_greater_returns = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_greater_returns.fit(df_scaled, y_greater_returns)
    df['Return_Potential'] = rf_greater_returns.predict(df_scaled)
    stocks_data['greaterreturn'] = {
        "title": "Top 5 Stocks for Greater Returns",
        "stocks": df[['Name', 'Return_Potential', 'CMP Rs.', 'ROE %', 'P/E']]
        .sort_values(by='Return_Potential', ascending=False)
        .head(5)
        .to_dict(orient='records'),
        "metric": "Return_Potential"
    }

    app.run(debug=True)
