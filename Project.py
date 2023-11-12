###############################################################################
# FINANCIAL PROGRAMMING PROJECT
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from plotly.subplots import make_subplots

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
#==============================================================================
# Header
#==============================================================================

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 2 selection boxes to select: Ticker and Time Period
    """
    
    # Add dashboard title 
    st.header('My Financial Dashboard')
    
    # Ticker list 
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    # Defining global variables
    global ticker 
    global end_date 
    global start_date
    
    #Creating dropdow select boxes
    with st.container():
        ticker= st.selectbox('Enter a stock ticker symbol (e.g., AAPL):', ticker_list,index=24)
        #AMZN ticker is set by default
        time_period = st.selectbox('Select time period:', ['1 month', '6 months', 'Year to Date', '1 year', '5 years','MAX'],index=3)   
        #1 year is set as default value
        
    #Defining current date as end date
    end_date = datetime.now()
    
    #Defining start date depending on the time_period select box
    if time_period == '1 month':
        start_date = (datetime.now() - timedelta(days=30))
    elif time_period == '6 months':
        start_date = (datetime.now() - timedelta(days=180))
    elif time_period == 'Year to Date':
        start_date = datetime(datetime.now().year, 1, 1)
    elif time_period == '1 year':
        start_date = (datetime.now() - timedelta(days=365))
    elif time_period == '5 years':
        start_date = (datetime.now() - timedelta(days=1825))
    elif time_period == 'MAX':
        start_date = end_date - pd.DateOffset(years= 60)
        
    def update ():
        stock_data= yf.download(ticker)
        csv_filename = f"{ticker}_data.csv"
        stock_data.to_csv(csv_filename, index=False)
    
    if st.button("Update and Download"):
        update()
#==============================================================================
# Tab 1
#==============================================================================
def render_tab1():
    """
    This function render the Tab 1 - Company Profile of the dashboard.
    """
       
    # Create two columns to show the two tables side to side
    col1, col2= st.columns([1, 1])
    
    # Get the company information
    @st.cache_data
    
    #Function to get company information:
    def Finfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """        
        return YFinance(ticker).info
    
    # Function to get ticker values based on specific time period
    def ystock(ticker, start_date, end_date):
        stock = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        #Building the dataframe to update second table 
        #Values are based on time period selection
        previous_close = stock['Close'].values[0]
        open_price = stock['Open'].values[0]
        high_price = stock['High'].values[0]
        low_price = stock['Low'].values[0]
        volume = stock['Volume'].values[0]
        avg_volume = stock['Volume'].mean()
        
        # Construct a list of dictionaries for each row
        summarystock = [{'Previous Close': previous_close, 'Open': open_price, 'High': high_price, 'Low': low_price, 'Volume': volume, 'Avg. Volume': avg_volume}]
        # Create a dataframe from the dictionary and transpose vertically
        df = pd.DataFrame(summarystock).T
        return df
    
    # Function to get ticker values and just keep 'Close'
    def ystock2(ticker, start_date, end_date):
        stock2 = yf.Ticker(ticker).history(start=start_date, end=end_date)
        df2 = stock2['Close']
        return df2 
    
    #Make sure a ticker is selected  
    if ticker != '':
    #However, on the selectbox AMZN has been set by default     
        
        with st.sidebar:
        # Get the company information in list format
            info = Finfo(ticker)
            
            # Show the company description using markdown + HTML
            st.write('###### Company Summary:')
            st.markdown('<div style="text-align: justify; font-size: 12;">' + \
                        info['longBusinessSummary'] + \
                        '</div><br>',
                        unsafe_allow_html=True)
                
            st.write('###### Mqjor Stakeholders:')
            # Get the company major stakeholders
            major_holders=yf.Ticker(ticker).get_major_holders()
            #Putting the information in html table format to easily customize
            table_html = major_holders.to_html(index=False, header=False, classes=["no-header-table"])
            styled_html = f'<div style="font-size: 10px;">{table_html}</div>'
            st.markdown(styled_html, unsafe_allow_html=True)
                
        
        with col1:
            #This column will show static current company stock info
            info = Finfo(ticker)
            st.write('##### Current Statistics:')
            info_keys = {'previousClose':'Previous Close',
                         'open'         :'Open',
                         'bid'          :'Bid',
                         'ask'          :'Ask',
                         'marketCap'    :'Market Cap',
                         'volume'       :'Volume'}
            #Dictionary
            company_stats = {} 
            for key in info_keys:
                company_stats.update({info_keys[key]:info[key]})
            #Convert to DataFrame
            company_stats = pd.DataFrame({'Value':pd.Series(company_stats)}) 
            #Putting the information in html table format to easily customize
            table_html = company_stats.to_html(header=False, classes=["no-header-table"],float_format='%.2f')
            styled_html = f'<div style="font-size: 12.5px;">{table_html}</div>'
            st.markdown(styled_html, unsafe_allow_html=True)
        
        with col2:
            #This column will show company stock info back from the time period seleted
            df=ystock(ticker, start_date, end_date)
            st.write('##### Compare to your selected period:')
            #Putting the information in html table format to easily customize
            table_html = df.to_html(header=False, classes=["no-header-table"],float_format='%.2f')
            styled_html = f'<div style="font-size: 12.5px;">{table_html}</div>'
            st.markdown(styled_html, unsafe_allow_html=True)
            
        #Outside of the column, the below     
        df2=ystock2(ticker, start_date, end_date)
        fig = go.Figure()
        #Adding a fill to be an area chart under the line
        fig.add_trace(go.Scatter(x=df2.index, y=df2, fill='tozeroy', fillcolor='rgba(200,230,250,0.2)'))
        
        #Adding cosmetic changes to the graph
        #One notable is making the background transparent and no grids
        fig.update_layout(
            xaxis=dict(showgrid=False), 
            yaxis=dict(showgrid=False),  
            yaxis_title="Close Price (in USD)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            )
        fig.update_xaxes(titlefont=dict(size=12))
        fig.update_yaxes(titlefont=dict(size=12))
        
        #Final plot
        st.plotly_chart(fig)
    
 #==============================================================================
 # Tab 2
 #==============================================================================  
def render_tab2():
     
     #Creating a selectbox to select the chart type
     #Making Candlestick the default
     chart_type = st.selectbox('Select chart type:', ['Line', 'Candlestick'],index=1)
     
     #Creating a selectbox to select the interval type
     global interval_type
     interval_mapping = {
     '1d': '1 Day',
     '1wk': '1 Week',
     '1mo': '1 Month' }
     interval_type = st.selectbox('Select interval:', list(interval_mapping.keys()),index=0)
     #Daily interval is made by default
     
     @st.cache_data
     # Function to get ticker values with interval type included in parameters 
     def ystock(ticker, start_date, end_date, interval_type):
        stock = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval_type)
        df = stock
        return df
     
    # When selecting chart type as a line, do:
     if chart_type == 'Line':
         df= ystock(ticker, start_date, end_date,interval_type)
         #Calculates moving average
         df['MA50'] = df['Close'].rolling(window=len(df),min_periods=1).mean()
         
         #Plotting
         fig = go.Figure()
         #Prepare the plot to for having two yaxis
         fig = make_subplots(specs=[[{"secondary_y": True}]])
         #Adding trace for line
         fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name= 'Close Price'))
         #Adding trace for volume
         fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',yaxis='y2',opacity=0.5))
         #Adding trace for moving average
         fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-Day Moving Average'))
         
         #Adding cosmetic customization to the plot
         plt.grid(False)
         fig.update_layout(
             xaxis=dict(showgrid=False,tickfont=dict(size=10)),  # Remove x-axis grid lines
             yaxis=dict(showgrid=False,tickfont=dict(size=10)),  # Remove y-axis grid lines,
             plot_bgcolor="rgba(0,0,0,0)",
             paper_bgcolor="rgba(0,0,0,0)",
             )
             
         fig.update_yaxes(title_text="Close Price", secondary_y=False)
         fig.update_yaxes(title_text="Volume", secondary_y=True)
         
         #Line Plot
         st.plotly_chart(fig)
     
    # When selecting chart type as a line, do:
    # This is the default chart; no scenario where selection is empty
     if chart_type == 'Candlestick':
         df= ystock(ticker, start_date, end_date,interval_type)
         #Calculate moving average
         df['MA50'] = df['Close'].rolling(window=len(df),min_periods=1).mean()
         #Prepare the plot to for having two yaxis
         fig = make_subplots(specs=[[{"secondary_y": True}]])
         #Adding trace for the candlestick
         fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
         #Adding trace for volume
         fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',yaxis='y2',opacity=0.4))
         #Adding trace for moving average
         fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-Day Moving Average'))
         
         #Adding cosmetic customization to the plot
         plt.grid(False)
         fig.update_layout(
             xaxis=dict(showgrid=False, tickfont=dict(size=10)),
             yaxis=dict(showgrid=False, tickfont=dict(size=10)),
             plot_bgcolor="rgba(0,0,0,0)",
             paper_bgcolor="rgba(0,0,0,0)",
         )
         
         #Candlestick Plot
         st.plotly_chart(fig)
                  
#==============================================================================
# Tab 3
#==============================================================================
def render_tab3():
    
    #Number of simulations selectbox
    financial_type = st.selectbox('Select financial statement:', ['Balance Sheet', 'Cashflow','Income Statement'],index=0)
    #Balance sheet is made the option by default
    period_type = st.selectbox('Select period:', ['Quaterly', 'Annual'],index=0)
    #Quaterly is made the option by default
    #There will always be values for selection since select boxes have default options.
    
    
    #To match the selection above to attributes that yf.Ticker understands to fecth the statements
    if financial_type == 'Balance Sheet' and period_type == 'Quaterly':
        statement=yf.Ticker(ticker).quarterly_balance_sheet
    if financial_type == 'Balance Sheet' and period_type == 'Annual':
        statement=yf.Ticker(ticker).balance_sheet
    if financial_type == 'Cashflow' and period_type == 'Quaterly':
        statement=yf.Ticker(ticker).quarterly_cashflow
    if financial_type == 'Cashflow' and period_type == 'Annual':
        statement=yf.Ticker(ticker).cashflow
    if financial_type == 'Income Statement' and period_type == 'Quaterly':
       statement=yf.Ticker(ticker).quarterly_income_stmt
    if financial_type == 'Income Statement' and period_type == 'Annual':
       statement=yf.Ticker(ticker).income_stmt

    #Function to get the financial statement
    def finance():
        #The try/except to capture errors when the statements are not available on yfinance
        try:
            #For statements that are not available
            if statement is not None:
                st.dataframe(statement)
        except AttributeError:
            st.warning('Data is not available for this ticket in this period. Select another ticket and period combination')
            
    finance()   
    
#==============================================================================
# Tab 4
#==============================================================================
def render_tab4():
   
    #Number of simulations selectbox
    nsimulations = st.selectbox('Select Number of Simulations:', [200, 500, 1000])
    #Horizon selectbox
    thorizon = st.selectbox('Select time horizon in days:', [30, 60, 90])
    
    @st.cache_data
    #Declaring MonteCarlo Function
    def MonteCarlo(ticker, start_date, end_date, nsimulations, thorizon):

        mstock = yf.Ticker(ticker).history(start=start_date, end=end_date)
        mclose = mstock['Close']
        daily_return = mclose.pct_change()
        daily_volatility = np.std(daily_return)
        mlast = mclose[-1]
        next_price = []
        
        #Passing thorizon parameter
        for n in range(thorizon):
            
            future_return = np.random.normal(0, daily_volatility)
            future_price = mlast * (1 + future_return)
            next_price.append(future_price)
            mlast = future_price
        
        #Keeping the seed for this exercise
        np.random.seed(123)
        
        simulation_df = pd.DataFrame()

        #Passing parameter nsimulations
        for i in range(nsimulations):
            next_price = []
            mlast = mclose[-1]
    
            for j in range(thorizon):
            # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)
    
            # Generate the random future price
                future_price = mlast * (1 + future_return)
    
            # Save the price and go next
                next_price.append(future_price)
                mlast = future_price
        
        # Store the result of the simulation
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
            
            ending_price = simulation_df.iloc[-1:, :].values[0, ]
            future_price_95ci = np.percentile(ending_price, 5)
            VaR = mclose[-1] - future_price_95ci
        
        #Printing the result of VaR
        st.write("###### The value of risk (VaR) is:")
        st.subheader(VaR)

        #Plotting the figure
        fig, ax = plt.subplots(figsize=(15, 10))
        #Make the background transparent
        ax.patch.set_alpha(0)  
        ax.plot(simulation_df)
        ax.axhline(y=mclose[-1], color='red')
        ax.set_xlabel('Day', color='grey')
        ax.set_ylabel('Price', color='grey')
        ax.xaxis.label.set_color('grey')
        ax.yaxis.label.set_color('grey')
        ax.tick_params(axis='x', colors='grey')
        ax.tick_params(axis='y', colors='grey')
        ax.legend(['Current stock price is: ' + str(np.round(mclose[-1], 2))])
        ax.get_legend().legend_handles[0].set_color('red')
        fig.patch.set_alpha(0)       
        return fig
                  
        st.pyplot(fig)
        
    fig= MonteCarlo(ticker, start_date, end_date, nsimulations, thorizon)
    st.pyplot(fig)
    
#==============================================================================
# Tab 5
#==============================================================================
def render_tab5():
    st.subheader('What is your safest investment?')
    
    # Creating a multiselection ticker selectbox to compare stocks
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    selected_tickers = st.multiselect("Select up to 3 tickers to compare your initial choice:", ticker_list)
    
    # Putting a try statement to prevent any errors from selecting more than 3 additional tickers
    try: 
        #Adding an if for limiting comparison to 3 stocks plus the initially selected
        if len(selected_tickers) < 4:
            #Invest function will calculate VaR at 95%
            def Invest(ticker, start_date, end_date, nsim, thor):
                mstock = yf.Ticker(ticker).history(start=start_date, end=end_date)
                mclose = mstock['Close']
                daily_return = mclose.pct_change()
                daily_volatility = np.std(daily_return)
                mlast = mclose[-1]
                next_price = []
                
                for n in range(thor):
                    future_return = np.random.normal(0, daily_volatility)
                    future_price = mlast * (1 + future_return)
                    next_price.append(future_price)
                    mlast = future_price
                
                np.random.seed(123)
                simulation_df = pd.DataFrame()
            
                for i in range(nsim):
                    next_price = []
                    mlast = mclose[-1]
            
                    for j in range(thor):
                        future_return = np.random.normal(0, daily_volatility)
                        future_price = mlast * (1 + future_return)
                        next_price.append(future_price)
                        mlast = future_price
                
                    next_price_df = pd.Series(next_price).rename('sim' + str(i))
                    simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
                    
                ending_price = simulation_df.iloc[-1:, :].values[0, ]
                future_price_95ci = np.percentile(ending_price, 5)
                VaR = mclose[-1] - future_price_95ci
                return VaR

            # Initialize my plot
            fig = go.Figure()

            # Initial VaR from initial stock
            VaR= Invest(ticker, start_date, end_date, 1000, 90)
           
            #Creating Min variables to compare later on the code with the other tickers
            MinVar= VaR
            MinTicker= ticker
            
            #Plotting the VaR from the initial ticker
            fig.add_trace(go.Bar(x=['Initial VaR'], y=[VaR], opacity=0.7, name=ticker))

            # Calculating VaR for selected additional tickers
            # Creating a for loop to go through all the nez selected tickers
            # For each selected ticker run the Invest Function and calculate VaR
            for i, selected_ticker in enumerate(selected_tickers, start=1):
                VaR = Invest(selected_ticker, start_date, end_date, 1000, 90)
                if VaR < MinVar:
                    #If the new calculate VaR is smaller than the one before, save in MinVar
                    MinVar=VaR
                    #If the new calculate VaR is smaller than the one before, save the ticker in MinTicker
                    MinTicker=selected_ticker
                
                # Add VaR bar for all other tickers
                fig.add_trace(go.Bar(x=[f'VaR{i}'], y=[VaR], opacity=0.7, name=selected_ticker))
            
            #Write which ticker had the smallest VaR
            st.write(f"The safest investment is {MinTicker} with a value of risk at 95% confidence of: {MinVar:.2f} USD", color='blue')
            
            #Add cosmetic updates to the plot; like making background transparent
            fig.update_layout(xaxis=dict(ticks='',showticklabels=False))
            fig.update_layout(xaxis_showgrid=False)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig)
        
    except AttributeError:
        st.write('You can select up to 3 tickers to compare only')
        st.warning('You can select up to 3 tickers to compare only')
#==============================================================================
# Main body
#==============================================================================
      
# Render the header
render_header()

# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Charts","Finance","MonteCarlo Simulation","Invest"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()

    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #1d1936;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
###############################################################################
# END
###############################################################################
