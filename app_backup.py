from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import traceback
import numpy as np
from datetime import datetime, timedelta
import calendar
from collections import defaultdict
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for communication with Streamlit

# Global DataFrame to hold uploaded data
df_global = None

# Predefined categories
PREDEFINED_CATEGORIES = [
    'Groceries', 'Utilities', 'Rent', 'Entertainment', 'Transportation',
    'Dining', 'Shopping', 'Healthcare', 'Education', 'Insurance',
    'Investment', 'Travel', 'Personal Care', 'Home & Garden', 'Other'
]

def categorize_transaction(description):
    """Categorize transaction based on description"""
    description = str(description).upper()
    
    # Keyword-based categorization logic (this part is correct)
    if any(keyword in description for keyword in ['GROCERY', 'SUPERMARKET', 'FOOD', 'VEGETABLE', 'FRUIT', 'MILK', 'BREAD', 'RICE', 'DAL', 'OIL', 'SPICE', 'KIRANA', 'GENERAL STORE', 'BIG BAZAAR', 'RELIANCE FRESH', 'DMART', 'GROFERS', 'BIGBASKET']):
        return 'Groceries'
    elif any(keyword in description for keyword in ['ELECTRICITY', 'POWER', 'GAS', 'WATER', 'INTERNET', 'PHONE', 'MOBILE', 'BROADBAND', 'WIFI', 'UTILITY', 'BILL', 'PAYMENT', 'BSNL', 'AIRTEL', 'JIO', 'VODAFONE', 'IDEA', 'MTNL']):
        return 'Utilities'
    elif any(keyword in description for keyword in ['RENT', 'HOUSE RENT', 'ACCOMMODATION', 'LEASE', 'RENTAL']):
        return 'Rent'
    elif any(keyword in description for keyword in ['MOVIE', 'CINEMA', 'NETFLIX', 'AMAZON PRIME', 'HOTSTAR', 'ENTERTAINMENT', 'GAME', 'GAMING', 'PLAYSTATION', 'XBOX', 'NINTENDO', 'BOOK', 'MAGAZINE', 'NEWSPAPER', 'MUSIC', 'SPOTIFY', 'YOUTUBE', 'STREAMING']):
        return 'Entertainment'
    elif any(keyword in description for keyword in ['PETROL', 'DIESEL', 'FUEL', 'GAS', 'UBER', 'OLA', 'TAXI', 'BUS', 'TRAIN', 'METRO', 'PARKING', 'TOLL', 'TRANSPORT', 'CAB', 'AUTO', 'PETROL PUMP', 'HP', 'SHELL', 'BP', 'INDIAN OIL']):
        return 'Transportation'
    elif any(keyword in description for keyword in ['RESTAURANT', 'CAFE', 'FOOD', 'MEAL', 'LUNCH', 'DINNER', 'BREAKFAST', 'SWIGGY', 'ZOMATO', 'FOODPANDA', 'DOMINOS', 'PIZZA HUT', 'KFC', 'MCDONALDS', 'SUBWAY', 'CAFETERIA', 'CANTEEN', 'HOTEL', 'BAR', 'PUB']):
        return 'Dining'
    elif any(keyword in description for keyword in ['AMAZON', 'FLIPKART', 'MYNTRA', 'SHOPPING', 'PURCHASE', 'MALL', 'SHOP', 'RETAIL', 'CLOTHING', 'FASHION', 'SHOES', 'ELECTRONICS', 'APPLIANCES', 'FURNITURE', 'DECOR', 'LIFESTYLE', 'JABONG', 'SNAPDEAL', 'PAYTM MALL', 'TATA CLIQ', 'NYKAA', 'LENSKART']):
        return 'Shopping'
    elif any(keyword in description for keyword in ['HOSPITAL', 'DOCTOR', 'MEDICAL', 'PHARMACY', 'MEDICINE', 'HEALTH', 'CLINIC', 'DENTAL', 'SURGERY', 'AMBULANCE', 'APOLLO', 'FORTIS', 'MAX HOSPITAL', 'MEDPLUS', 'NETMEDS', 'PRACTO', 'HEALTHKART']):
        return 'Healthcare'
    elif any(keyword in description for keyword in ['SCHOOL', 'COLLEGE', 'UNIVERSITY', 'EDUCATION', 'TUITION', 'FEES', 'COURSE', 'TRAINING', 'BOOKS', 'LIBRARY', 'EXAM', 'BYJU', 'UNACADEMY', 'VEDANTU', 'STUDENT', 'ACADEMIC']):
        return 'Education'
    elif any(keyword in description for keyword in ['INSURANCE', 'POLICY', 'PREMIUM', 'LIC', 'HDFC LIFE', 'ICICI PRU', 'SBI LIFE', 'BAJAJ ALLIANZ', 'TATA AIG', 'RELIANCE GENERAL', 'HEALTH INSURANCE', 'MOTOR INSURANCE', 'TERM INSURANCE']):
        return 'Insurance'
    elif any(keyword in description for keyword in ['MUTUAL FUND', 'SIP', 'INVESTMENT', 'TRADING', 'ZERODHA', 'GROWW', 'ANGEL BROKING', 'UPSTOX', 'PAYTM MONEY', 'KUVERA', 'STOCK', 'EQUITY', 'BOND', 'FD', 'RD', 'PPF', 'ELSS', 'NSE', 'BSE']):
        return 'Investment'
    elif any(keyword in description for keyword in ['IRCTC', 'MAKEMYTRIP', 'GOIBIBO', 'CLEARTRIP', 'YATRA', 'TRAVEL', 'BOOKING', 'HOTEL', 'FLIGHT', 'TRAIN', 'BUS', 'TICKET', 'VACATION', 'HOLIDAY', 'TOURISM', 'AIRBNB', 'OYO', 'TREEBO', 'REDBUS']):
        return 'Travel'
    elif any(keyword in description for keyword in ['SALON', 'PARLOUR', 'BEAUTY', 'COSMETICS', 'SKINCARE', 'HAIRCUT', 'MASSAGE', 'SPA', 'WELLNESS', 'FITNESS', 'GYM', 'YOGA', 'PERSONAL CARE', 'GROOMING', 'URBAN COMPANY', 'LAKME']):
        return 'Personal Care'
    elif any(keyword in description for keyword in ['HOME DEPOT', 'GARDEN', 'PLANTS', 'NURSERY', 'HARDWARE', 'TOOLS', 'REPAIR', 'MAINTENANCE', 'PLUMBER', 'ELECTRICIAN', 'CARPENTER', 'PAINT', 'TILES', 'CEMENT', 'CONSTRUCTION', 'RENOVATION']):
        return 'Home & Garden'
    else:
        return 'Other'

# ==============================================================================
# ✨ NEW HELPER FUNCTION TO PROCESS ANY DATAFRAME ✨
# ==============================================================================
def process_dataframe(df):
    """Cleans, processes, and categorizes a transaction dataframe."""
    if df.empty:
        raise ValueError("The provided CSV data is empty.")

    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle different CSV formats
    if 'Withdrawal Amt.' in df.columns and 'Deposit Amt.' in df.columns:
        df['Withdrawal Amt.'] = pd.to_numeric(df['Withdrawal Amt.'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['Deposit Amt.'] = pd.to_numeric(df['Deposit Amt.'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['Amount'] = df['Deposit Amt.'] - df['Withdrawal Amt.']
        if 'Narration' in df.columns:
            df['Description'] = df['Narration'].fillna('No description')
        else:
            df['Description'] = 'No description'
    elif 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(',', ''), errors='coerce')
        if 'Description' not in df.columns:
            if 'Narration' in df.columns:
                df['Description'] = df['Narration'].fillna('No description')
            else:
                df['Description'] = 'No description'
    else:
        raise ValueError("CSV must contain either 'Amount' or both 'Withdrawal Amt.' and 'Deposit Amt.' columns.")

    if 'Date' not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    # Process and add required columns
    df['Category'] = df['Description'].apply(categorize_transaction)
    df['custom_name'] = ''
    df = df.dropna(subset=['Amount'])
    df['id'] = range(1, len(df) + 1)
    
    return df

@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    global df_global
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        df = pd.read_csv(file)
        df_global = process_dataframe(df) # Use the new helper function
        
        return jsonify({
            "message": "File processed successfully",
            "total_transactions": len(df_global),
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# ==============================================================================
# ✨ THIS IS THE NEW, CORRECTED ROUTE FOR SAMPLE DATA ✨
# ==============================================================================
@app.route('/api/get_sample_data', methods=['GET'])
def get_sample_data():
    global df_global
    try:
        # Make sure 'sample_data.csv.csv' is in your GitHub repository
        df = pd.read_csv('sample_data.csv.csv')
        df_global = process_dataframe(df) # Use the new helper function
        
        return jsonify({
            'message': 'Sample data processed successfully',
            'total_transactions': len(df_global)
        })
    except FileNotFoundError:
        return jsonify({'error': 'The sample_data.csv.csv file was not found on the server.'}), 404
    except Exception as e:
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

# Your other routes remain the same...

@app.route("/api/get_transactions", methods=["GET"])
def get_transactions():
    global df_global
    if df_global is None:
        return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
    try:
        display_columns = ['id', 'Date', 'Description', 'Amount', 'Category', 'custom_name']
        available_columns = [col for col in display_columns if col in df_global.columns]
        data = df_global[available_columns].to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error retrieving transactions: {str(e)}"}), 500

@app.route("/api/get_other_transactions", methods=["GET"])
def get_other_transactions():
    global df_global
    if df_global is None:
        return jsonify({"error": "No data available"}), 400
    try:
        other_txns = df_global[df_global['Category'] == 'Other']
        display_columns = ['id', 'Date', 'Description', 'Amount', 'Category', 'custom_name']
        available_columns = [col for col in display_columns if col in other_txns.columns]
        data = other_txns[available_columns].to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error retrieving other transactions: {str(e)}"}), 500

@app.route("/api/update_category", methods=["POST"])
def update_category():
    global df_global
    if df_global is None:
        return jsonify({"error": "No data available"}), 400
    try:
        data = request.get_json()
        transaction_id = int(data.get('id'))
        new_category = data.get('category')
        custom_name = data.get('custom_name', '')
        
        mask = df_global['id'] == transaction_id
        if not mask.any():
            return jsonify({"error": f"Transaction ID {transaction_id} not found."}), 404
        
        df_global.loc[mask, 'Category'] = new_category
        df_global.loc[mask, 'custom_name'] = custom_name
        return jsonify({"message": "Category updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error updating category: {str(e)}"}), 500

@app.route("/api/add_custom_category", methods=["POST"])
def add_custom_category():
    global df_global
    if df_global is None:
        return jsonify({"error": "No data available"}), 400
    try:
        data = request.get_json()
        transaction_id = int(data.get('id'))
        custom_category = data.get('custom_category')
        description_keywords = data.get('description_keywords', [])
        
        mask = df_global['id'] == transaction_id
        if not mask.any():
            return jsonify({"error": f"Transaction ID {transaction_id} not found."}), 404
        
        df_global.loc[mask, 'Category'] = custom_category
        df_global.loc[mask, 'custom_name'] = custom_category
        
        if description_keywords:
            for keyword in description_keywords:
                if keyword.strip():
                    keyword_mask = df_global['Description'].str.contains(keyword.strip(), case=False, na=False)
                    df_global.loc[keyword_mask, 'Category'] = custom_category
                    df_global.loc[keyword_mask, 'custom_name'] = custom_category
        
        return jsonify({"message": f"Custom category '{custom_category}' added successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error adding custom category: {str(e)}"}), 500

@app.route("/api/get_expense_summary", methods=["GET"])
def get_expense_summary():
    global df_global
    if df_global is None:
        return jsonify({"error": "No data available"}), 400
    try:
        expenses = df_global[df_global['Amount'] < 0].copy()
        if expenses.empty:
            return jsonify([]) # Return empty list if no expenses
        
        expenses['Amount'] = expenses['Amount'].abs()
        expenses['Display_Category'] = expenses.apply(lambda row: row['custom_name'] if row['custom_name'] else row['Category'], axis=1)
        
        summary = expenses.groupby('Display_Category').agg({'Amount': 'sum', 'id': 'count'}).reset_index()
        summary.columns = ['Category', 'Amount', 'Transaction_Count']
        summary = summary.sort_values('Amount', ascending=False)
        summary['Amount'] = summary['Amount'].round(2)
        
        return jsonify(summary.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": f"Error creating expense summary: {str(e)}"}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Expense Analyzer API is running"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5001)

