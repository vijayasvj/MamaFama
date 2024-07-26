import os
import subprocess
import sys

def install_dependencies():
    if not os.path.exists(os.path.expanduser("~/miniconda")):
        subprocess.check_call(["bash", "setup.sh"])

install_dependencies()

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import scipy.optimize as opt

# Function to calculate MAMA and FAMA using TA-Lib
def calculate_mama_fama(stock_data, fast_limit, slow_limit):
    import talib
    # Validate parameters
    if not (0.01 <= fast_limit <= 0.99) or not (0.01 <= slow_limit <= 0.99):
        raise ValueError("fast_limit and slow_limit must be between 0.01 and 0.99")
    
    # Validate stock data
    if len(stock_data) < 10:
        raise ValueError("Insufficient data for TA-Lib calculation")
    
    # Calculate MAMA and FAMA values using the given fast and slow limits
    mama, fama = talib.MAMA(stock_data, fastlimit=fast_limit, slowlimit=slow_limit)
    return mama, fama  # Return the calculated MAMA and FAMA values

# Function to perform backtesting and calculate profit or loss without involving money
def backtest_mama_fama(df, fast_limit, slow_limit):
    try:
        # Calculate MAMA and FAMA values for the stock data
        df['MAMA'], df['FAMA'] = calculate_mama_fama(df['Close'].values, fast_limit, slow_limit)
        df.dropna(inplace=True)  # Drop any rows with NaN values to clean the data

        actions = []  # List to store actions taken during backtesting
        holding = False  # Boolean flag to indicate if holding a stock
        buy_price = 0  # Price at which the stock was bought
        total_return = 0  # Total profit percentage

        # Loop through the data to simulate trading based on MAMA and FAMA values
        for i in range(1, len(df) - 1):
            if df['MAMA'].iloc[i] > df['FAMA'].iloc[i] and df['MAMA'].iloc[i - 1] <= df['FAMA'].iloc[i - 1]:
                # Buy signal
                if not holding:
                    buy_price = df['Close'].iloc[i + 1]
                    actions.append(f'BUY at {df["Close"].iloc[i + 1]}')
                    holding = True
            elif df['MAMA'].iloc[i] < df['FAMA'].iloc[i] and df['MAMA'].iloc[i - 1] >= df['FAMA'].iloc[i - 1]:
                # Sell signal
                if holding:
                    sell_price = df['Close'].iloc[i + 1]
                    profit_percentage = ((sell_price - buy_price) / buy_price) * 100
                    total_return += profit_percentage
                    actions.append(f'SELL at {df["Close"].iloc[i + 1]} (Profit: {profit_percentage:.2f}%)')
                    holding = False
            else:
                actions.append('HOLD')  # If neither condition is met, hold

        # If there's an open position, close it at the last closing price
        if holding:
            sell_price = df['Close'].iloc[-1]
            profit_percentage = ((sell_price - buy_price) / buy_price) * 100
            total_return += profit_percentage
            actions.append(f'SELL at {df["Close"].iloc[-1]} (final trade) (Profit: {profit_percentage:.2f}%)')

        return total_return, actions  # Return total profit percentage and list of actions

    except ValueError as e:
        st.error(f"Error: {e}")
        return 0, []

# Objective function to be maximized
def objective(params, df):
    a, b = params
    try:
        overall_return, _ = backtest_mama_fama(df.copy(), a, b)
        return -overall_return  # Negative because we want to maximize
    except Exception as e:
        st.error(f"Optimization error: {e}")
        return float('inf')  # Return a large number if there's an error

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Visualise backtesting", "Optimise with Yfinance data", "Upload CSV and optimise"])

# Home Page
if page == "Visualise backtesting":
    # App title
    st.title("Stock Market Analysis with MAMA and FAMA")

    # Introduction
    st.markdown(
        """
        <div class="intro-text">
        <p>Analyze and optimize stock trading strategies using MAMA and FAMA indicators. Enter a stock symbol, select a date range, and optimize the parameters to maximize your trading returns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input fields for stock analysis
    st.header("Input Stock Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        stock_symbol = st.text_input("Stock Symbol", value="AAPL")
        
    with col2:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        
    with col3:
        end_date = st.date_input("End Date", value=pd.to_datetime("2020-12-31"))

    st.header("MAMA and FAMA Parameters")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown('<p class="slider-label">Fast Limit</p>', unsafe_allow_html=True)
        fast_limit = st.slider("Fast Limit", min_value=0.01, max_value=0.99, value=0.5, step=0.01, label_visibility="collapsed")
        
    with col5:
        st.markdown('<p class="slider-label">Slow Limit</p>', unsafe_allow_html=True)
        slow_limit = st.slider("Slow Limit", min_value=0.01, max_value=0.99, value=0.05, step=0.01, label_visibility="collapsed")

    # Fetch historical data
    if stock_symbol:
        try:
            df = yf.download(stock_symbol, start=start_date, end=end_date)
            if not df.empty:
                df['Close'] = df['Adj Close']
                # Calculate MAMA and FAMA using the given limits
                try:
                    df['MAMA'], df['FAMA'] = calculate_mama_fama(df['Close'].values, fast_limit, slow_limit)
                    
                    # Plotting with Plotly
                    st.subheader("Stock Price with MAMA and FAMA")
                    fig = go.Figure()
                    # Add trace for closing price
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='#0073e6')))
                    # Add trace for MAMA
                    fig.add_trace(go.Scatter(x=df.index, y=df['MAMA'], mode='lines', name='MAMA', line=dict(color='#FF6347')))
                    # Add trace for FAMA
                    fig.add_trace(go.Scatter(x=df.index, y=df['FAMA'], mode='lines', name='FAMA', line=dict(color='#4682B4')))
                    fig.update_layout(
                        title=dict(text=f"{stock_symbol} Stock Price with MAMA and FAMA", font=dict(color='white')),
                        xaxis_title=dict(text='Date', font=dict(color='white')),
                        yaxis_title=dict(text='Price', font=dict(color='white')),
                        xaxis=dict(tickfont=dict(color='white')),
                        yaxis=dict(tickfont=dict(color='white')),
                        plot_bgcolor='#1e1e1e', 
                        paper_bgcolor='#1e1e1e'
                    )
                    st.plotly_chart(fig, use_container_width=True)  # Display the plot
                    
                except ValueError as e:
                    st.error(f"Calculation error: {e}")

            re, ac = backtest_mama_fama(df, fast_limit, slow_limit)
            st.subheader(f"Returns: {re}%")
        except KeyError as e:
            st.error(f"Error fetching data for {stock_symbol}: {e}")

# About Page
elif page == "Optimise with Yfinance data":
    # Section for Optimization and Walk-Forward Testing
    st.header("Optimization and Walk-Forward Testing")
        # Input fields for stock analysis
    st.header("Input Stock Data")
    col6, col7, col22 = st.columns(3)

    with col6:
        stock_symbol = st.text_input("Stock Symbol", value="AAPL")

    with col7:
        opt_start_date = st.date_input("Optimization Start Date", value=pd.to_datetime("2020-01-01"))
        
    with col22:
        opt_end_date = st.date_input("Optimization End Date", value=pd.to_datetime("2020-12-31"))

    # Optimization section
    st.subheader("Optimize MAMA and FAMA Parameters")

    if 'optimal_params' not in st.session_state:
        st.session_state.optimal_params = None
        st.session_state.returns = 0
        st.session_state.actions = []

    if st.button("Optimize Parameters"):
        try:
            # Filter data for the optimization date range
            opt_df = yf.download(stock_symbol, start=opt_start_date, end=opt_end_date)
            if not opt_df.empty:
                opt_df['Close'] = opt_df['Adj Close']
            if not opt_df.empty:
                st.session_state.opt_df = opt_df
                # Bounds for a and b
                bounds = [(0.01, 0.99), (0.01, 0.99)]

                # Use differential evolution to find the maximum
                result = opt.differential_evolution(objective, bounds, args=(opt_df,), seed=0)

                # Extract the optimal parameters
                st.session_state.optimal_params = result.x
                # Perform backtesting with optimized parameters
                st.session_state.returns, st.session_state.actions = backtest_mama_fama(opt_df, st.session_state.optimal_params[0], st.session_state.optimal_params[1])
            else:
                st.error("No data available for the selected optimization date range.")
        except KeyError as e:
            st.error(f"Error fetching data for optimization: {e}")

    # Walk-Forward Testing Section
    if st.session_state.optimal_params is not None:
        optimal_a, optimal_b = st.session_state.optimal_params
        st.write(f"Optimal Fast Limit: {optimal_a}, Optimal Slow Limit: {optimal_b}")
        st.write(f"The profits we have got by buying and selling a single stock is: {st.session_state.returns}%")
        actions_df = pd.DataFrame({'Date': st.session_state.opt_df.index[:len(st.session_state.actions)], 'Action': st.session_state.actions})
        actions_csv = actions_df.to_csv(index=False)  # Convert the DataFrame to CSV format
        st.download_button(label="Download Optimized Actions CSV", data=actions_csv, file_name='optimized_actions.csv', mime='text/csv')
        st.subheader("Walk-Forward Testing")
        col10, col11 = st.columns(2)
        st.write("Use these optimised parameters and do a walk forward testing!")
        with col10:
            st.markdown('<p class="slider-label">Fast Limit</p>', unsafe_allow_html=True)
            fast_limit = st.slider("Fast Limit", min_value=0.01, max_value=0.99, value=optimal_a, step=0.01, label_visibility="collapsed")
            
        with col11:
            st.markdown('<p class="slider-label">Slow Limit</p>', unsafe_allow_html=True)
            slow_limit = st.slider("Slow Limit", min_value=0.01, max_value=0.99, value=optimal_b, step=0.01, label_visibility="collapsed")
        col8, col9 = st.columns(2)

        with col8:
            wf_start_date = st.date_input("Walk-Forward Start Date", value=pd.to_datetime("2021-01-01"))
        
        with col9:
            wf_end_date = st.date_input("Walk-Forward End Date", value=pd.to_datetime("2021-12-31"))
        
        # Button to perform walk-forward testing
        if st.button("Perform Walk-Forward Testing"):
            try:
                wf_df = yf.download(stock_symbol, start=pd.to_datetime(wf_start_date), end=pd.to_datetime(wf_end_date))
                if not wf_df.empty:
                    wf_df['Close'] = wf_df['Adj Close']
                    # Calculate MAMA and FAMA using the given limits
                    try:
                        optimal_a2 = float(f"{optimal_a}")
                        optimal_b2 = float(f"{optimal_b}")
                        wf_return, wf_actions = backtest_mama_fama(wf_df, optimal_a2, optimal_b2)
                        st.write(f"Walk-Forward Testing Return: {wf_return}%")

                        # Display actions taken during walk-forward testing
                        st.write("Actions taken during Walk-Forward Testing:")
                        wf_actions_df = pd.DataFrame({'Date': wf_df.index[:len(wf_actions)], 'Action': wf_actions})
                        wf_actions_csv = wf_actions_df.to_csv(index=False)  # Convert the DataFrame to CSV format
                        st.download_button(label="Download Walk-Forward Actions CSV", data=wf_actions_csv, file_name='walkforward_actions.csv', mime='text/csv')
                        wf_df['MAMA'], wf_df['FAMA'] = calculate_mama_fama(wf_df['Close'].values, fast_limit, slow_limit)
                        # Separate columns for the graphs
                        st.subheader("Graphical Representation of Optimization and Walk-Forward Testing")
                        col12, col13 = st.columns(2)

                        with col12:
                            st.write("Optimization Timeframe")
                            fig_opt = go.Figure()
                            # Add trace for optimization phase closing price
                            fig_opt.add_trace(go.Scatter(x=st.session_state.opt_df.index, y=st.session_state.opt_df['Close'], mode='lines', name='Optimization Close Price', line=dict(color='#0073e6')))
                            # Add trace for MAMA
                            fig_opt.add_trace(go.Scatter(x=st.session_state.opt_df.index, y=st.session_state.opt_df['MAMA'], mode='lines', name='MAMA', line=dict(color='#FF6347')))
                            # Add trace for FAMA
                            fig_opt.add_trace(go.Scatter(x=st.session_state.opt_df.index, y=st.session_state.opt_df['FAMA'], mode='lines', name='FAMA', line=dict(color='#4682B4')))
                            fig_opt.update_layout(
                                title=dict(text=f"{stock_symbol} Optimization Timeframe", font=dict(color='white')),
                                xaxis_title=dict(text='Date', font=dict(color='white')),
                                yaxis_title=dict(text='Price', font=dict(color='white')),
                                xaxis=dict(tickfont=dict(color='white')),
                                yaxis=dict(tickfont=dict(color='white')),
                                plot_bgcolor='#1e1e1e', 
                                paper_bgcolor='#1e1e1e'
                            )
                            st.plotly_chart(fig_opt, use_container_width=True)

                        with col13:
                            st.write("Walk-Forward Testing Timeframe")
                            fig_wf = go.Figure()
                            # Add trace for walk-forward phase closing price
                            fig_wf.add_trace(go.Scatter(x=wf_df.index, y=wf_df['Close'], mode='lines', name='Walk-Forward Close Price', line=dict(color='#FF7F50')))
                            # Add trace for MAMA
                            fig_wf.add_trace(go.Scatter(x=wf_df.index, y=wf_df['MAMA'], mode='lines', name='MAMA', line=dict(color='#FF6347')))
                            # Add trace for FAMA
                            fig_wf.add_trace(go.Scatter(x=wf_df.index, y=wf_df['FAMA'], mode='lines', name='FAMA', line=dict(color='#4682B4')))
                            fig_wf.update_layout(
                                title=dict(text=f"{stock_symbol} Walk-Forward Testing Timeframe", font=dict(color='white')),
                                xaxis_title=dict(text='Date', font=dict(color='white')),
                                yaxis_title=dict(text='Price', font=dict(color='white')),
                                xaxis=dict(tickfont=dict(color='white')),
                                yaxis=dict(tickfont=dict(color='white')),
                                plot_bgcolor='#1e1e1e', 
                                paper_bgcolor='#1e1e1e'
                            )
                            st.plotly_chart(fig_wf, use_container_width=True)

                    except ValueError as e:
                        st.error(f"Calculation error: {e}")
                else:
                    st.error("No data available for the selected walk-forward testing date range.")
            except KeyError as e:
                st.error(f"Error fetching data for walk-forward testing: {e}")

# Contact Page
elif page == "Upload CSV and optimise":
    st.title("Upload CSV and Optimise")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=True, index_col='Date')
        
        if 'Close' not in df.columns:
            st.error("CSV file must contain a 'Close' column.")
        else:
            st.write("CSV file successfully loaded. Here are the first few rows:")
            st.dataframe(df.head())
            
            st.header("Optimization Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                opt_start_date = st.date_input("Optimization Start Date", value=df.index.min())
                
            with col2:
                opt_end_date = st.date_input("Optimization End Date", value=df.index.min() + pd.DateOffset(years=1))
                
            if opt_start_date < df.index.min() or opt_end_date > df.index.max():
                st.error("The specified date range is not present in the CSV.")
            else:
                st.subheader("Optimize MAMA and FAMA Parameters")
                
                if 'optimal_params' not in st.session_state:
                    st.session_state.optimal_params = None
                    st.session_state.returns = 0
                    st.session_state.actions = []
                
                if st.button("Optimize Parameters"):
                    opt_df = df[(df.index >= opt_start_date) & (df.index <= opt_end_date)]
                    
                    if not opt_df.empty:
                        st.session_state.opt_df = opt_df
                        bounds = [(0.01, 0.99), (0.01, 0.99)]
                        
                        result = opt.differential_evolution(objective, bounds, args=(opt_df,), seed=0)
                        
                        st.session_state.optimal_params = result.x
                        st.session_state.returns, st.session_state.actions = backtest_mama_fama(opt_df, st.session_state.optimal_params[0], st.session_state.optimal_params[1])
                        
                        st.write(f"Optimal Fast Limit: {st.session_state.optimal_params[0]}, Optimal Slow Limit: {st.session_state.optimal_params[1]}")
                        st.write(f"Optimization Return: {st.session_state.returns}%")
                        
                        actions_df = pd.DataFrame({'Date': st.session_state.opt_df.index[:len(st.session_state.actions)], 'Action': st.session_state.actions})
                        actions_csv = actions_df.to_csv(index=False)
                        st.download_button(label="Download Optimized Actions CSV", data=actions_csv, file_name='optimized_actions.csv', mime='text/csv')
                
                if st.session_state.optimal_params is not None:
                    st.subheader("Walk-Forward Testing")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        wf_start_date = st.date_input("Walk-Forward Start Date", value=opt_end_date + pd.DateOffset(days=1))
                        
                    with col4:
                        wf_end_date = st.date_input("Walk-Forward End Date", value=wf_start_date + pd.DateOffset(years=1))
                        
                    if wf_start_date < df.index.min() or wf_end_date > df.index.max():
                        st.error("The specified date range is not present in the CSV.")
                    else:
                        if st.button("Perform Walk-Forward Testing"):
                            wf_df = df[(df.index >= wf_start_date) & (df.index <= wf_end_date)]
                            
                            if not wf_df.empty:
                                wf_return, wf_actions = backtest_mama_fama(wf_df, st.session_state.optimal_params[0], st.session_state.optimal_params[1])
                                st.write(f"Walk-Forward Testing Return: {wf_return}%")
                                
                                wf_actions_df = pd.DataFrame({'Date': wf_df.index[:len(wf_actions)], 'Action': wf_actions})
                                wf_actions_csv = wf_actions_df.to_csv(index=False)
                                st.download_button(label="Download Walk-Forward Actions CSV", data=wf_actions_csv, file_name='walkforward_actions.csv', mime='text/csv')
                                
                                wf_df['MAMA'], wf_df['FAMA'] = calculate_mama_fama(wf_df['Close'].values, st.session_state.optimal_params[0], st.session_state.optimal_params[1])
                                
                                st.subheader("Graphical Representation of Walk-Forward Testing")
                                fig_wf = go.Figure()
                                fig_wf.add_trace(go.Scatter(x=wf_df.index, y=wf_df['Close'], mode='lines', name='Walk-Forward Close Price', line=dict(color='#FF7F50')))
                                fig_wf.add_trace(go.Scatter(x=wf_df.index, y=wf_df['MAMA'], mode='lines', name='MAMA', line=dict(color='#FF6347')))
                                fig_wf.add_trace(go.Scatter(x=wf_df.index, y=wf_df['FAMA'], mode='lines', name='FAMA', line=dict(color='#4682B4')))
                                fig_wf.update_layout(
                                    title=dict(text=f"Walk-Forward Testing Timeframe", font=dict(color='white')),
                                    xaxis_title=dict(text='Date', font=dict(color='white')),
                                    yaxis_title=dict(text='Price', font=dict(color='white')),
                                    xaxis=dict(tickfont=dict(color='white')),
                                    yaxis=dict(tickfont=dict(color='white')),
                                    plot_bgcolor='#1e1e1e', 
                                    paper_bgcolor='#1e1e1e'
                                )
                                st.plotly_chart(fig_wf, use_container_width=True)
                            else:
                                st.error("No data available for the selected walk-forward testing date range.")
