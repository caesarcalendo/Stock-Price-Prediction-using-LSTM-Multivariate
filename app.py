from flask import Flask, render_template, request
import yfinance as yf 
import pandas as pd 
import numpy as np
from datetime import date, timedelta, datetime 
from sklearn.preprocessing import MinMaxScaler 
from model import LSTM_encoder_decoder, evaluate_model
import plotly
import plotly.graph_objects as go 
import plotly.express as px
import json
import math

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

# Get datetime in business days exclude weekend Saturday and Sunday
def date_by_adding_business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date


# For get Close Price in the last element of list
def getLastValue(aList):
    return aList[-1]

# For round up decimal up to 10
def roundup(x):
    return int(math.ceil(x / 5.0)) * 5


@app.route('/predict', methods=['GET','POST'])
def predict():
    global data_emittan, kode_emitten, periode_waktu, ohlc_graph, future_graph, n_input_steps, n_output_steps
    global mse_score, mad_score, mape_score, date_pred, price_pred
    if request.method == 'POST':
        # Input form for prediction
        kode_emitten = request.form['kode_emitten']
        kode_emitten = str(kode_emitten) + ".JK"
        print(kode_emitten)
        periode_waktu = int(request.form['periode_waktu'])
        
        # Get LQ45 stock data from yahoo finance
        end_date =  date.today().strftime("%Y-%m-%d")
        start_date = '2017-09-19'  
        df = yf.download(kode_emitten, start=start_date, end=end_date)
        df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

        #========================== 5 SAMPLE LAST DATA OHLC ===============================#
        df_5_sample = df.copy()
        df_5_sample.reset_index(inplace=True)
        df_5_sample['Date'] = pd.to_datetime(df_5_sample['Date'])
        df_5_sample = df_5_sample[['Date', 'Open', 'High', 'Low', 'Close']].tail(5)
        data_emittan = pd.DataFrame(df_5_sample)
        print(data_emittan)

        #========================== APPLY MODELING, EVALUATION & PREDICTION ===============================#        
        # Picking the multivariate series 
        variables = ['Open','High','Low','Close']
        data_modeling = df[variables] 

        # Pisahkan data pelatihan menjadi set data pelatihan dan latih
        # Sebagai langkah pertama, dapatkan jumlah baris untuk melatih model pada data 80%
        train_ratio = 0.8
        train_data_length = int(len(data_modeling)*train_ratio)

        # Buat data pelatihan dan pengujian
        train_data = data_modeling[:train_data_length]
        test_data = data_modeling[train_data_length:]

        # Normalization
        sc = MinMaxScaler()
        train = sc.fit_transform(train_data)
        test = sc.transform(test_data)
        print(train.shape,test.shape)

        # LSTM Encoder Decoder Hyperparameters
        n_output_steps = periode_waktu  # Number of outputs we want to predict into the future
        n_input_steps = periode_waktu   # Number of past inputs that we want to use to predict the future
        neuron = 128
        lr = 1e-3
        batch_size = 128
        epochs = 100

        model_en_dec, history_en_dec = LSTM_encoder_decoder(train,n_output_steps,n_input_steps,neuron,lr,batch_size,epochs) 

        # Evaluating the model
        mse_score, mad_score, mape_score, true_X, true_Y, predicted = evaluate_model(model_en_dec, test, n_output_steps, n_input_steps)
        print('MSE = {}'.format(mse_score))
        print('MAD = {}'.format(mad_score))
        print('MAPE = {}'.format(mape_score))

        if periode_waktu == 3:
            # Get each date of training data and testing data
            train_date = train_data.index
            test_date = test_data.index[2 + n_output_steps:]

            # Bring back into original values
            train_original = sc.inverse_transform(train)
            test_original = sc.inverse_transform(test)

            # Create empty fill use numpy zeros, so predicted data fits with scaler dimension
            zeros_shape = test.shape[0] - n_input_steps - 2
            num_x_variables = len(variables) - 1
            empty_fill = np.zeros((zeros_shape, num_x_variables))

            # Transform back into original value
            lstm_pred = np.concatenate((empty_fill, predicted), axis=1)
            lstm_pred = sc.inverse_transform(lstm_pred)
        
        else:
            # Get each date of training data and testing data
            train_date = train_data.index
            test_date = test_data.index[4 + n_output_steps:]

            # Bring back into original values
            train_original = sc.inverse_transform(train)
            test_original = sc.inverse_transform(test)

            # Create empty fill use numpy zeros, so predicted data fits with scaler dimension
            zeros_shape = test.shape[0] - n_input_steps - 4
            num_x_variables = len(variables) - 1
            empty_fill = np.zeros((zeros_shape, num_x_variables))

            # Transform back into original value
            lstm_pred = np.concatenate((empty_fill, predicted), axis=1)
            lstm_pred = sc.inverse_transform(lstm_pred)


        future_dicts = []

        forecast_date = df.index[-1]

        for i in range(periode_waktu):
            forecast_date = date_by_adding_business_days(forecast_date, 1)
            num_x_variables = len(variables) - 1
            empty_fill = np.zeros((periode_waktu, num_x_variables))
        
            future_pred = model_en_dec.predict(true_X[-periode_waktu:], verbose=0)
            predictions_future = np.concatenate((empty_fill, future_pred), axis=1)
            predictions_future = sc.inverse_transform(predictions_future)
            future_dicts.append({'Predictions':predictions_future[i], "Date": forecast_date })

        future_days = pd.DataFrame(future_dicts).set_index("Date")
        future_days['Predictions'] = future_days['Predictions'].apply(getLastValue)
        future_days['Predictions'] = future_days['Predictions'].apply(roundup)
        print(future_days)

        result_df = pd.DataFrame(future_dicts)
        date_pred = result_df['Date']
        date_pred = date_pred.dt.date
        price_pred = np.round(future_days['Predictions'],6)

        #========================== OHLC + FUTURE NEXT DAYS PREDICTION PLOT ===============================#

        trace0 = go.Candlestick(x=data_modeling.index,
                                open=data_modeling['Open'], 
                                high=data_modeling['High'],
                                low=data_modeling['Low'], 
                                close=data_modeling['Close'])

        trace1 = go.Scatter(
            x = test_data.index,
            y = test_original[:, -1],
            name = 'Actual Close Price'
        )
        trace2 = go.Scatter(
            x = test_date,
            y = lstm_pred[:, -1],
            name = 'Prediction Close Price'
        )
        trace3 = go.Scatter(
            x = future_days.index,
            y = future_days['Predictions'],
            name = 'Future Close Price'
        )
        layout = go.Layout(
            title = f'OHLC Stock Close Price {kode_emitten} Prediction with LSTM Encoder Decoder',
            xaxis = {'title' : "Date"},
            yaxis = {'title' : "Close Price"}
        )
        fig = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)
        ohlc_graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        #========================== FUTURE NEXT DAYS PREDICTION PLOT ===============================#

        test_original = df[-n_input_steps:]

        new_pred_plot = pd.DataFrame({
            'last_original_close_value':test_original['Close'],
            'next_predicted_close_value':future_days['Predictions'],
        })

        fig2 = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_close_value'],
                                                            new_pred_plot['next_predicted_close_value']],
                    labels={'value': 'Stock price','index': 'Date'})
        fig2.update_layout(title_text=f'Compare last {periode_waktu} days vs next {periode_waktu} days',
                        plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=False)
        future_graph = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('prediksi.html',  
                            mse=mse_score,
                            mad=mad_score,
                            mape=mape_score,
                            date_pred=date_pred,
                            price_pred=price_pred,
                            column_names=data_emittan.columns.values, 
                            row_data=list(data_emittan.values.tolist()), 
                            zip=zip,
                            ohlc_graph=ohlc_graph,
                            future_graph=future_graph
                            )


if __name__ == "__main__":
    app.run(debug=True)