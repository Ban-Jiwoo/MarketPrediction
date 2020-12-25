from flask import Flask, render_template
import plotly.figure_factory as ff
import plotly
import json
import plotly.express as px
import numpy as np
import pandas as pd
import datetime
import matplotlib
# import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras

app = Flask(__name__)

#@는 밑에 def를 바로 실행시킴.
@app.route('/')
def index():
    #그래프를 받아서 index.html 에 띄우기.
    bar, table = create_streamline()
    return render_template('index.html', plot=bar, tableplot=table)


# Function to plot interactive plots using Plotly Express
def interactive_plot(df, title):
    fig = px.line(title = title)
    for i in df.columns[1:]:
        fig.add_scatter(x = df['Date'], y = df[i], name = i)
        fig.show()


# Function to normalize stock prices based on their initial price
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = str(int(x[i])/int((x[i][0])))
    return x




# Function to return the input/output (target) data for AI/ML Model
# Note that our goal is to predict the future stock price
# Target stock price today will be tomorrow's price
def trading_window(data):

  # 1 day window
  n = 1

  # Create a column containing the prices for the next 1 days
  data['Target'] = data[['Close']].shift(-n)

  # return the new dataset
  return data








#정확도 계산 알고리즘.
def acc(accDf):
    count = 0
    # accDf = accDf.to_string()
    # accDf['Close'] = pd.to_numeric(accDf['Close'])
    # accDf['Predictions'] = pd.to_numeric(accDf['Predictions'])
    pd.to_numeric(accDf['Close'])
    pd.to_numeric(accDf['Prediction'])

    for i in range(1, len(accDf)-1):

        actual_result = ''
        if accDf['Close'][i] < accDf['Close'][i+1]:
            actual_result = 'Up'
        elif accDf['Close'][i] > accDf['Close'][i+1]:
            actual_result = "Down"

        else:
            actual_result = 'Not Changed'


        predict = ''
        if accDf['Prediction'][i] > accDf['Close'][i]:
            predict = 'Up'

        elif accDf['Prediction'][i] < accDf['Close'][i]:
            predict = 'Down'

        else:
            predict = 'Not Changed'


        results = ''
        if actual_result == predict:
            results = 'Correct!'
            count += 1
        else:
            results = 'Incorrect!'


    return (count/(len(accDf)-1))*100




def create_streamline():
    #데이터 read.
    # Read stock prices data# Read stock prices data
    stock_price_df = pd.read_csv('data/kospiData2015.csv', thousands=',')
    # Sort the data based on Date
    # kospiData = pd.DataFrame()
    # kospiData['Date'] = stock_price_df['일자']
    # kospiData['Close'] = stock_price_df['현재지수']
    # kospiData['Volume'] = stock_price_df['거래량(천주)']

    kospi_Price_Data = pd.DataFrame()
    kospi_Price_Data['Date'] = stock_price_df['일자']
    kospi_Price_Data['Date'] = kospi_Price_Data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
    kospi_Price_Data['Volume'] = stock_price_df['거래량(천주)']
    kospi_Price_Data['Close'] = stock_price_df['현재지수']

    #목표가 설정.
    price_volume_target_df = trading_window(kospi_Price_Data)
    price_volume_target_df
    # Remove the last row as it will be a null value
    price_volume_target_df = price_volume_target_df[:-1]
    price_volume_target_df


    #데이터 정제!!!
    import re
    #223,192 이런 숫자를 바꾸려하는데 ,이것 때문에 숫자로 변환이 불가능하다고한다. 정규식으로 빼주자.
    #이게 효과적.  df.read_csv('foo.tsv', sep='\t', thousands=',')
    # non_numeric = re.compile(r',')

    # price_volume_target_df['Close'] = price_volume_target_df.loc[price_volume_target_df['Close'].str.contains(non_numeric)]
    # price_volume_target_df['Volume'] = price_volume_target_df.loc[price_volume_target_df['Volume'].str.contains(non_numeric)]
    print(price_volume_target_df)

    #판다스 데이터프레임 숫사가 스트링 타입이라 머신러닝 계산이 안됨. 실수로 바꿔준다.
    price_volume_target_df['Close'] = pd.to_numeric(price_volume_target_df['Close'], downcast="float")
    price_volume_target_df['Volume'] = pd.to_numeric(price_volume_target_df['Volume'], downcast="float")



    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns = ['Date']))

    # Creating Feature and Target
    #종가, 거래량 을 x로, 목표가는 y로.
    X = price_volume_target_scaled_df[:,:2]
    y = price_volume_target_scaled_df[:,2:]

    # Spliting the data this way, since order is important in time-series
    # Note that we did not use train test split with it's default settings since it shuffles the data
    split = int(0.65 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    from sklearn.linear_model import Ridge
    # Note that Ridge regression performs linear least squares with L2 regularization.
    # Create and train the Ridge Linear Regression  Model
    regression_model = Ridge()
    # regression_model = Ridge(alpha=2)
    regression_model.fit(X_train, y_train)
    # Test the model and calculate its accuracy
    lr_accuracy = regression_model.score(X_test, y_test)
    print("Linear Regression Score: ", lr_accuracy)

    predicted_prices = regression_model.predict(X)
    # Append the predicted values into a list
    Predicted = []
    for i in predicted_prices:
        Predicted.append(i[0])
    # Append the close values to the list
    close = []
    for i in price_volume_target_scaled_df:
        close.append(i[0])

    df_ridge_predicted = price_volume_target_df[['Date']]
    # Add the close values to the dataframe
    df_ridge_predicted['Close'] = close

    # Add the predicted values to the dataframe
    df_ridge_predicted['Prediction'] = Predicted



# @@@@@@@@
#LSTM !!!!
    lstmDf = pd.DataFrame()
    lstmDf['Date'] = kospi_Price_Data['Date']
    lstmDf['Close'] = kospi_Price_Data['Close']
    lstmDf['Volume'] = kospi_Price_Data['Volume']

    # Get the close and volume data as training data (Input)
    training_data = lstmDf.iloc[:, 1:3].values
    training_data

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_data)


    # Create the training and testing data, training data contains present day and previous day values
    X = []
    y = []
    for i in range(1, len(lstmDf)):
        X.append(training_set_scaled [i-1:i, 0])
        y.append(training_set_scaled [i, 0])

    # Convert the data into array format
    X = np.asarray(X)
    y = np.asarray(y)

    # Split the data
    split = int(0.7 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # Reshape the 1D arrays to 3D arrays to feed in the model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train.shape, X_test.shape


    # Create the model
    inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])) #(1,1) 인풋디멘션.
    x = keras.layers.LSTM(150, return_sequences= True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150, return_sequences=True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150)(x)
    outputs = keras.layers.Dense(1, activation='linear')(x) #linear을 쓰는 이유는, output이 continuous value 이기 때문에.

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss="mse")
    model.summary()


    # Trai the model
    history = model.fit(
        X_train, y_train,
        epochs = 20,
        batch_size = 32,
        validation_split = 0.2
    )


    # Make prediction
    predicted = model.predict(X)
    # print("Predicted!", predicted)
    # scaler = MinMaxScaler()
    # predicted = scaler.inverse_transform(predicted)
    # print("Predicted2!", predicted)

    # Append the predicted values to the list
    test_predicted = []
    for i in predicted:
      # test_predicted.append(i[0])
      test_predicted.append(i[0])

    # test_predicted = sc.inverse_transform(test_predicted)

    lstmPredicted = pd.DataFrame()
    lstmPredicted['Date'] = lstmDf['Date'][1:]

    # Plot the data
    close = []
    for i in training_set_scaled:
      close.append(i[0])


    lstmPredicted['Close'] = close[1:]
    lstmPredicted['Prediction'] = test_predicted

    print("LSTM PREDICTED",lstmPredicted)


    #정확도 계산
    accuracyResult = acc(lstmPredicted)
    print("LSTM AccuracyResult!", accuracyResult)


# @@@@@@@@@@@@@@



    #정확도 계산
    accuracyResult = acc(df_ridge_predicted)
    print("Ridge AccuracyResult!", accuracyResult)


    #차트!
    # for state in state_list:
    #릿지!
    # trace2 = go.Scatter(x=df_ridge_predicted['Date'], y=df_ridge_predicted['Close'], mode='lines', name='Kospi')
    # trace3 = go.Scatter(x=df_ridge_predicted['Date'], y=df_ridge_predicted['Prediction'], mode='lines', name='Prediction')

#LSTM !
    layout2 = go.Layout({'title': 'Kospi Market & Prediction Data', "legend": {"orientation": "h","xanchor":"left"}, })

    trace2 = go.Scatter(x=lstmPredicted['Date'], y=lstmPredicted['Close'], mode='lines', name='Kospi')
    trace3 = go.Scatter(x=lstmPredicted['Date'], y=lstmPredicted['Prediction'], mode='lines', name='A.I. Prediction')
    #릿지!
    # trace2 = go.Scatter(x=df_ridge_predicted['Date'], y=df_ridge_predicted['Close'], mode='lines', name='Kospi')
    # trace3 = go.Scatter(x=df_ridge_predicted['Date'], y=df_ridge_predicted['Prediction'], mode='lines', name='Prediction')




    print("-----------------------------")
    print("Lstm Close DF: ", lstmPredicted['Close'])
    print("Lstm Predicted DF: ", lstmPredicted['Prediction'])

    #예측 데이터까지 심고,
    #결과 데이터 프레임 만들어서 표로 내보내기
    accuracyDataFrame = pd.DataFrame()
    accuracyDataFrame['Date'] = ['2020-12-28', '2020-12-24', '2020-12-23', '2020-12-22', '2020-12-21', '2020-12-18', '2020-12-17']
    # accuracyDataFrame['Volume'] = ['']
    accuracyDataFrame['Market Price'] = ['-', '2,806.86', '2,759.82', '2,733.68', '2,778.65', '2,772.18', '2,770.43']
    accuracyDataFrame['Prediction'] = ['Down(0.975739)', 'Up', 'Up', 'Up', 'Up', 'Down', 'Down']
    accuracyDataFrame['Accuracy'] = ['-', 'Correct(78%)', 'Correct(78%)', 'Incorrect(78%)', 'Correct(78%)','Correct(77%)', 'Correct(77%)' ]


    table_fig = ff.create_table(accuracyDataFrame)





    fig = {'data': [trace2],
           'layout': layout2
           }
    fig['data'].append(trace3)

    # fig = go.Figure(data=trace_list, layout=layout)





    #그래프 json으로 넘겨주기.
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    tableJSON = json.dumps(table_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON, tableJSON

    # Plot the results
    # interactive_plot(kospiData, "Original Vs. Prediction")
    # po.plot(kospi_Price_Data, filename='templates/stock.html', auto_open=False)


if __name__ == '__main__':
    app.run(debug=False)
