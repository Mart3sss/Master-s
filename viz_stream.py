import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup


import locale


def main_locale():
    locale.setlocale(locale.LC_NUMERIC, '')
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # Ajusta a tu configuración regional si es diferente



def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return components.drop('CIK', axis=1).set_index('Symbol')


def load_quotes(ticker, period = '3y', interval = '1d'):
    #eliminar nas
    
    return yf.download(ticker, period= period, interval=interval)

 
key_statistics_dic = {}
def load_statics(ticker):

        #scraping key statistics
    url= 'https://finance.yahoo.com/quote/{}/key-statistics?p={}'.format(ticker, ticker)
    key_statistics_dic[ticker] = {}
        
    headers = {'User-Agent': 'Chrome/117.0.5938.150'}
    page = requests.get(url, headers = headers)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table", {'class': "W(100%) Bdcl(c)"})
        
    for t in tabl:
            rows = t.find_all("tr")
            for row in rows:
                if len(row.get_text(separator='|').split("|")[0:2])>0:
                    key_statistics_dic[ticker][row.get_text(separator='|').split("|")[0]]=row.get_text(separator='|').split("|")[-1]    
        
    key_statistics_dic[ticker] = pd.DataFrame(key_statistics_dic[ticker], index = [0]).T
        # Luego de crear el DataFrame key_statistics_dic[ticker], puedes aplicar fillna('')
    key_statistics_dic[ticker] = key_statistics_dic[ticker].fillna(' ')

    return key_statistics_dic[ticker]

def ATR(DF, n=14):
    df = DF.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = df['High'] - df['Adj Close'].shift(1)
    df['L-PC'] = df['High'] - df['Adj Close'].shift(1)
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].ewm(com=n, min_periods=n).mean()
    return df['ATR']

def ADX(DF, n=20):
    df = DF.copy()
    df['ATR'] = ATR(df, n)
    df['upmove'] = df['High'] - df['High'].shift(1)
    df['downmove'] = df['Low'].shift(1) - df['Low']
    df['+dm'] = np.where((df['upmove'] > df['downmove']) & (df['upmove'] > 0), df['upmove'], 0)
    df['-dm'] = np.where((df['downmove'] > df['upmove']) & (df['downmove'] > 0), df['downmove'], 0)
    df['+di'] = 100 * (df['+dm'] / df['ATR']).ewm(span=n, min_periods=n).mean()
    df['-di'] = 100 * (df['-dm'] / df['ATR']).ewm(span=n, min_periods=n).mean()
    df['ADX'] = 100 * abs((df['+di'] - df['-di']) / (df['+di'] + df['-di'])).ewm(span=n, min_periods=n).mean()
    return df['ADX']

def Boll_Band(DF, n=14):
    df = DF.copy()
    df['MB'] = df['Adj Close'].rolling(n).mean()
    df['UB'] = df['MB'] + 2 * df['Adj Close'].rolling(n).std(ddof=0)
    df['LB'] = df['MB'] - 2 * df['Adj Close'].rolling(n).std(ddof=0)
    df['BB_Width'] = df["UB"] - df['LB']
    return df[['MB', 'UB', 'LB', 'BB_Width']]

def MACD(DF, a=12, b=26, c=9):
    df = DF.copy()
    df['ma_fast'] = df['Adj Close'].ewm(span=a, min_periods=a).mean()
    df['ma_slow'] = df['Adj Close'].ewm(span=b, min_periods=b).mean()
    df['macd'] = df['ma_fast'] - df['ma_slow']
    df['signal'] = df['macd'].ewm(span=c, min_periods=c).mean()
    return df.loc[:, ['macd', 'signal']]

def main():
    st.title("Análisis Técnico - Indicadores Financieros")

    st.sidebar.title("Options")
    show_companies_list = st.sidebar.checkbox('View companies list')
    
    if show_companies_list:
        components = load_data()
        st.subheader("Lista de Empresas")
        st.dataframe(components[['Security', 'GICS Sector', 'Date added', 'Founded']])

        st.sidebar.subheader('Select asset')
        asset = st.sidebar.selectbox('Click below to select a new asset',
                                     components.index.sort_values(), index=3,
                                     format_func=lambda x: x + ' - ' + components.loc[x].Security)
        st.title(components.loc[asset].Security)
        st.markdown(f"<h4 style='font-size: 14px; margin-top: -15px; margin-bottom: 5px;'>{components.loc[asset]['GICS Sub-Industry']}</h4>", unsafe_allow_html=True)  

        # Mostrar la información de la empresa sin el checkbox
        st.subheader("Información de la Empresa")
        st.table(components.loc[asset].iloc[1:])  # Excluir la primera fila

        # Obtener todos los tickers en la misma industria ########
        tickers_same_industry = components[components['GICS Sub-Industry'] == components.loc[asset]['GICS Sub-Industry']].index
        
        # Create a new dataframe with 'asset' and 'Security' columns
        asset_security_df = pd.DataFrame({'Asset': tickers_same_industry, 'Security': components.loc[tickers_same_industry, 'Security'].values})
        
        # Display the table with asset names and their corresponding securities
        st.subheader('Competidores de industría')
        st.markdown(asset_security_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
          

        data0 = load_quotes(asset)
        data = data0.copy().dropna()
        data.index.name = None


        # Sección para modificar el período y el intervalo de la búsqueda
        st.sidebar.subheader("Search Settings")
        period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        
        selected_period = st.sidebar.selectbox('Select Period', period_options, index=5)
        selected_interval = st.sidebar.selectbox('Select Interval', interval_options, index=3)

        data0 = load_quotes(asset, selected_period, selected_interval)
        data = data0.copy().dropna()
        data.index.name = None


        # Agregar la sección
        section = st.sidebar.slider('Number of quotes', min_value=30,
                                    max_value=min([2000, data.shape[0]]),
                                    value=500, step=10)

        data2 = data[-section:]['Adj Close'].to_frame('Adj Close')

    # Funciones Dinámicas - Medias Móviles
    st.sidebar.subheader('Moving Averages')
    
   
    # Media Móvil Simple (SMA)
    sma_expander = st.sidebar.expander("Simple Moving Average (SMA)")
    sma_enabled = sma_expander.checkbox('Enable SMA', value=True)
    if sma_enabled:
        period = sma_expander.slider('SMA period', min_value=5, max_value=500, value=20, step=1)
        data[f'SMA {period}'] = data['Adj Close'].rolling(period).mean()
        data2[f'SMA {period}'] = data[f'SMA {period}'].reindex(data2.index)
    
    # Media Móvil Simple 2 (SMA2)
    sma2_expander = st.sidebar.expander("Simple Moving Average 2 (SMA2)")
    sma2_enabled = sma2_expander.checkbox('Enable SMA2', value=True)
    if sma2_enabled:
        period2 = sma2_expander.slider('SMA2 period', min_value=5, max_value=500, value=100, step=1)
        data[f'SMA2 {period2}'] = data['Adj Close'].rolling(period2).mean()
        data2[f'SMA2 {period2}'] = data[f'SMA2 {period2}'].reindex(data2.index)
    
    # Bollinger Bands
    boll_band_expander = st.sidebar.expander("Bollinger Bands")
    enable_boll_band = boll_band_expander.checkbox('Enable Bollinger Bands', value=True)
    n_boll_band = boll_band_expander.slider('Bollinger Bands period', min_value=5, max_value=50, value=20, step=1)
    data[['MB', 'UB', 'LB', 'BB_Width']] = Boll_Band(data, n=n_boll_band)
    data2[['MB', 'UB', 'LB', 'BB_Width']] = data[['MB', 'UB', 'LB', 'BB_Width']].reindex(data2.index)
    
    # Gráfico
    st.subheader('Chart')
    
    # Convertir el índice a tipo Timestamp si es necesario
    data2.index = pd.to_datetime(data2.index)
    
    # Crear figura para los gráficos con subplots
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=[f'{asset} - Historical Data'])
    
    # Añadir medias móviles (SMA y SMA2)
    if sma_enabled:
        fig.add_trace(go.Scatter(x=data2.index, y=data2[f'SMA {period}'], mode='lines', name=f'SMA {period}', line=dict(width=1)))
    
    if sma2_enabled:
        fig.add_trace(go.Scatter(x=data2.index, y=data2[f'SMA2 {period2}'], mode='lines', name=f'SMA2 {period2}', line=dict(width=1)))
    
    # Añadir gráfico de Bollinger Bands
    if enable_boll_band:
        fig.add_trace(go.Scatter(x=data2.index, y=data2['MB'], mode='lines', name='MB', line=dict(width=1, color='rgba(255, 0, 0, 0.5)')))
        fig.add_trace(go.Scatter(x=data2.index, y=data2['UB'], mode='lines', name='UB', line=dict(width=1, color='rgba(0, 0, 255, 0.5)')))
        fig.add_trace(go.Scatter(x=data2.index, y=data2['LB'], mode='lines', name='LB', line=dict(width=1, color='rgba(0, 0, 255, 0.5)')))
        fig.add_trace(go.Scatter(x=data2.index, y=data2['UB'], fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', line=dict(width=0, color='rgba(0, 0, 255, 0.2)')))
    
    # Añadir gráfico de Precios
    fig.add_trace(go.Scatter(x=data2.index, y=data2['Adj Close'], mode='lines', name='Adj Close', line=dict(width=1)))
    
    # Configurar ejes
    fig.update_yaxes(title_text='Values', secondary_y=False)
    
    # Leyenda debajo del gráfico

    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y= 0, xanchor='auto', x=1))
    
    # Añadir grid
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'))
    

    # Mostrar el gráfico interactivo
    st.plotly_chart(fig)     
    
    # Funciones Dinámicas - Technical Indicators
    
    st.sidebar.subheader('Technical Indicators')
    
    # ADX
    adx_expander = st.sidebar.expander("Average True Range (ATR)")
    enable_adx = adx_expander.checkbox('Enable ADX', value=True)
    n_adx = adx_expander.slider('ADX period', min_value=5, max_value=50, value=20, step=1)
    data['ADX'] = ADX(data, n=n_adx)
    data2['ADX'] = data['ADX'].reindex(data2.index)
    

    
    # MACD
    macd_expander = st.sidebar.expander("Moving Average Convergence Divergence (MACD)")
    enable_macd = macd_expander.checkbox('Enable MACD', value=True)
    a_macd = macd_expander.slider('MACD fast period', min_value=5, max_value=50, value=12, step=1)
    b_macd = macd_expander.slider('MACD slow period', min_value=5, max_value=50, value=26, step=1)
    c_macd = macd_expander.slider('MACD signal period', min_value=5, max_value=50, value=9, step=1)
    data[['MACD', 'SIGNAL']] = MACD(data, a=a_macd, b=b_macd, c=c_macd)
    data2[['MACD', 'SIGNAL']] = data[['MACD', 'SIGNAL']].reindex(data2.index)
    
    # Gráfico ADX
    if enable_adx:
        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(x=data2.index, y=data2['ADX'], mode='lines', name='ADX'))
        fig_adx.update_layout(title='Average True Range (ADX)',
                              xaxis_title='Date', yaxis_title='ADX',
                              showlegend=True)
        st.plotly_chart(fig_adx)
    
    
    # Gráfico MACD
    if enable_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data2.index, y=data2['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data2.index, y=data2['SIGNAL'], mode='lines', name='Signal'))
        fig_macd.update_layout(title='Moving Average Convergence Divergence (MACD)',
                               xaxis_title='Date', yaxis_title='Values',
                               showlegend=True)
        st.plotly_chart(fig_macd)
  

    # Estadísticas de precios para la industria
    st.sidebar.subheader('Statistics for Industry')
    
    # Botón de selección para tipo de dato (Adj Close o Volumen)
    data_type = st.sidebar.radio("Select Data Type", ['Adj Close', 'Volumen'], index=0)
    
    
    ######## tener en cuenta que esto va fuera del grafico, porque las fechas estan estaticas
    ##### habria que establecer que fechas queremos mostrar en ambas partes.
    # Crear un DataFrame con los datos seleccionados para todas las acciones en la misma industria
    if data_type == 'Adj Close':
        data_industry = yf.download(tickers_same_industry.tolist(), period='3y', interval='1d')['Adj Close']
    else:
        data_industry = yf.download(tickers_same_industry.tolist(), period='3y', interval='1d')['Volume']
    
       # Calcular las estadísticas descriptivas
    industry_stats = data_industry.describe().round(2)
    
    # Mostrar la tabla con el nombre de la columna como el ticker y el título como estadísticas del tipo de dato seleccionado
    st.subheader(f'Industry {data_type} Statistics')
    st.table(industry_stats.T.style.format("{:n}").set_caption(f'Industry {data_type} Statistics'))
      
    # Main code
    key_statistics_data = {}
    
    # Load key statistics data for each ticker
    for ticker in tickers_same_industry:
        key_statistics_data[ticker] = load_statics(ticker)
    
    # Display key statistics in a single table
    st.subheader('Key Statistics')
    
    # Create an empty dataframe to store the concatenated data
    concatenated_data = pd.DataFrame()
    
    # Concatenate the key statistics data into a single dataframe
    for ticker in tickers_same_industry:
        # Add the data for each ticker to the concatenated dataframe
        concatenated_data[ticker] = key_statistics_data[ticker].squeeze()
    
    # Display the concatenated dataframe as a single table
    st.write(concatenated_data)
    
    
            
if __name__ == '__main__':
    main()

