from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    model = Sequential()
    # Adding more layers/complexity as typical for such tasks, 
    # ensuring it matches the 'LSTM Topology' if described, or standard practice.
    # The doc just says 'The architecture is designed...' let's stick to a standard one.
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(3, activation='softmax')) # 3 classes: Hold, Buy, Sell

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
