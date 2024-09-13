import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM


def train_models():
    print('===== Start Training Models =====')

    # 패턴 목록과 데이터 경로 설정
    pattern_list = ['ascending_triangle', 'ascending_wedge', 'descending_triangle', 'descending_wedge', 'double_top', 'double_bottom']
    data_path = "./algo_dataset"

    # 데이터를 담을 리스트
    data = []
    labels = []

    print('===== Loading Datas... =====')

    # 각 패턴별 데이터를 불러오기
    for pattern in pattern_list:
        for num in range(1, 10001):  # 각 패턴별로 1000개씩 있다고 가정
            file_path = os.path.join(data_path, pattern, f"{pattern}_{num}.csv")
            df = pd.read_csv(file_path)
            # OHLC 데이터만 사용한다고 가정 (Open, High, Low, Close)
            ohlc_data = df[['Open', 'High', 'Low', 'Close']].values
            
            # 고정된 길이로 패딩 (예: 길이를 100으로 고정)
            ohlc_data_padded = pad_sequences([ohlc_data], maxlen=100, padding='post', dtype='float32')[0]
            
            data.append(ohlc_data_padded)
            labels.append(pattern)

    print('===== Data Loaded! =====')

    # 데이터를 numpy 배열로 변환
    data = np.array(data)
    labels = np.array(labels)

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    # 데이터 스케일링
    scaler = StandardScaler()
    data_scaled = np.array([scaler.fit_transform(x) for x in data])

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels_categorical, test_size=0.2, random_state=42)

    # EarlyStopping 설정 (monitor='val_loss'를 기준으로, 5 에포크 동안 향상되지 않으면 중단)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print('===== Initialize CNN Model =====')

    # CNN 모델
    cnn_model = Sequential()

    cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(6, activation='softmax'))  # 6개의 클래스

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('===== CNN Model Training Started =====')

    # CNN 모델 학습
    cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    print('===== CNN Model Training Done! =====')

    print('===== Initialize LSTM Model =====')

    # LSTM 모델 생성
    lstm_model = Sequential()

    lstm_model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(LSTM(100))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(6, activation='softmax'))  # 6개의 클래스

    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('===== LSTM Model Training Started =====')

    # LSTM 모델 학습
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    print('===== LSTM Model Training Done! =====')

    print('''===== Sample Result of Model's Prediction''')

    # CNN 모델 예측
    cnn_predictions = cnn_model.predict(X_test)
    cnn_predicted_classes = np.argmax(cnn_predictions, axis=1)
    cnn_predicted_labels = label_encoder.inverse_transform(cnn_predicted_classes)

    # LSTM 모델 예측
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predicted_classes = np.argmax(lstm_predictions, axis=1)
    lstm_predicted_labels = label_encoder.inverse_transform(lstm_predicted_classes)

    # 예측 결과 출력
    for i in range(10):  # 테스트 샘플 10개에 대한 예측 결과를 출력
        print(f"Truth: {label_encoder.inverse_transform([np.argmax(y_test[i])])[0]}, CNN: {cnn_predicted_labels[i]}, LSTM: {lstm_predicted_labels[i]}")

    print('===== Initializing Save Directory =====')
    os.makedirs('save', exist_ok=True)

    # CNN 모델 저장 (Keras 네이티브 형식)
    cnn_model.save('./save/cnn_chart_pattern_model.keras')

    # LSTM 모델 저장 (Keras 네이티브 형식)
    lstm_model.save('./save/lstm_chart_pattern_model.keras')

    print('===== Model Saved! =====')