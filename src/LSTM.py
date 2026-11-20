import tensorflow as tf


def binary_cross_entropy(y_true, y_pred, eps=1e-7):
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


class LSTM:
    def __init__(self, sequence_length=10, input_dim=256, output_dim=1001,
                 hidden_units=128, dropout=0.2, learning_rate=1e-3):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = self._build()

    def _build(self):
        inputs = tf.keras.Input(shape=(self.sequence_length, self.input_dim), name='sequence_input')
        x = tf.keras.layers.LSTM(self.hidden_units, return_sequences=False,
                                 dropout=self.dropout, name='lstm')(inputs)
        outputs = tf.keras.layers.Dense(self.output_dim, activation='sigmoid', name='basket_output')(x)
        model = tf.keras.Model(inputs, outputs, name='next_basket_lstm')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy()
        )
        return model
