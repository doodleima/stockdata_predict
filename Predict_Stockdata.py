import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Model_definition import Stock_Model, dataset

def predict_stock(jongmok) :
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    label_cols = ['Close']

    df = pd.read_csv(".\\주가데이터\\" + str(jongmok) + " 주가추이.csv", encoding = "UTF-8-SIG")

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[scale_cols])

    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scale_cols

    SIZE = int(len(df_scaled) * 0.75) #int(len(normed_data) * 0.75)

    train = df_scaled[:SIZE]
    test = df_scaled[SIZE:]

    train_feature = train[feature_cols]
    train_label = train[label_cols]

    test_feature = test[feature_cols]
    test_label = test[label_cols]

    # Train / Test Dataset
    train_feature, train_label = dataset(train_feature, train_label, 31)
    test_feature, test_label = dataset(test_feature, test_label, 7)

    # Split Data : Train : 0.75 / Test : 0.25
    X_train, X_test, Y_train, Y_test = train_test_split(train_feature, train_label, test_size=0.25)

    erstoping = EarlyStopping(monitor = 'val_loss', patience = 5)
    ckpoint = ModelCheckpoint('.\\stock_model\\model.h5', monitor = 'val_loss', verbose = 1, save_best_only = True) #, mode = 'auto')

    model = Stock_Model(train_feature)

    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    #model.summary()

    hist = model.fit(X_train, Y_train, epochs=64, batch_size=16, validation_data=(X_test, Y_test), callbacks=[erstoping, ckpoint])
    predict = model.predict(test_feature)
    
    # plot : estimated result 
    """
    plt.figure(figsize=(9, 5))
    plt.plot(test_label, label='original')
    plt.plot(predict, label='predict')
    plt.legend()
    plt.show()
    """

    predict_week = predict[-7:]
    if predict_week[0] > predict_week[-1] :
        return "하락"
    elif predict_week[0] < predict_week[-1] :
        return "상승"
    else :
        return "동일"
