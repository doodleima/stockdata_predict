
def train_sentimentdata(jongmok) :
    import numpy as np

    from tensorflow.keras.models import load_model

    X_test = np.load(open('.\\word_dir\\test_data.npy', 'rb')) # 테스트 데이터 입력값
    Y_test = np.load(open('.\\word_dir\\test_label.npy', 'rb')) # 테스트 데이터 레이블

    print('테스트 정확도 : %.4f' % (load_model('model.h5').evaluate(X_test, Y_test)[1]))