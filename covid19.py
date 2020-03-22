import pandas as pd
from utils.data import setup_params,scale_data,prepare_data,create_sequences,split_data
from utils.plot import plot_losses,plot_data,plot_real_predicted,plot_predicted_data
from utils.predictions import predict_daily_cases
from net import CoronaVirusPredictor,train_model

# diff_daily_cases = prepare_data('time_series_19-covid-Confirmed.csv')
# seq_len = 3
# train_data, _, scaler = scale_data(diff_daily_cases,diff_daily_cases)
# X_train,y_train = create_sequences(train_data,seq_len)
# model = CoronaVirusPredictor(n_features=1,n_hidden=512,seq_len=seq_len,n_layers=2)
# model,train_hist,_ = train_model(model,X_train,y_train,num_epochs=120)
# DAYS_TO_PREDICT = 12
# predicted_cases,_ = predict_daily_cases(model,X_train,y_train,DAYS_TO_PREDICT,seq_len,scaler)
# predicted_cases = pd.Series(data=predicted_cases,
#     index=pd.date_range(start=diff_daily_cases.index[-1],
#                         periods=DAYS_TO_PREDICT + 1,
#                         closed='right'))
#
# plot_data(predicted_cases,'Predictions',label='Predicted Daily Cases')
# plot_real_predicted(diff_daily_cases,predicted_cases)

if __name__ == '__main__':
    setup_params()

    diff_daily_cases = prepare_data('time_series_19-covid-Confirmed.csv')
    train_data, test_data = split_data(diff_daily_cases, 20)
    train_data, test_data,scaler = scale_data(diff_daily_cases, train_data, test_data)
    seq_len = 5
    X_train, y_train = create_sequences(train_data, seq_len)
    X_test, y_test = create_sequences(test_data, seq_len)

    model = CoronaVirusPredictor(n_features=1, n_hidden=512, seq_len=seq_len, n_layers=2)

    model, train_hist, test_hist = train_model(model, X_train, y_train, X_test, y_test)

    plot_losses(train_hist, test_hist)

    predicted_cases, true_cases = predict_daily_cases(model, X_test, y_test, len(X_test), seq_len, scaler)

    plot_predicted_data(diff_daily_cases, train_data, true_cases, predicted_cases, scaler)