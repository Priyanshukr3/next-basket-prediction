import tensorflow as tf


def binary_cross_entropy(y_true, y_pred, eps=1e-7):
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


@tf.keras.utils.register_keras_serializable(package="nbp")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight=15.0, eps=1e-7, name="weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = float(pos_weight)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.eps, 1.0 - self.eps)
        loss = -(
            self.pos_weight * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pos_weight": self.pos_weight, "eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="nbp")
class RepeatGate(tf.keras.layers.Layer):
    """Learnable per-item scaling of a user's recency-weighted purchase history."""
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="repeat_alpha",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(1.5),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, history_freq):
        return history_freq * self.alpha


class LSTM:
    def __init__(
        self,
        sequence_length=10,
        input_dim=256,
        output_dim=1001,
        hidden_units=128,
        dropout=0.2,
        learning_rate=1e-3,
        pos_weight=15.0,
        use_history_residual=False,
    ):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight
        self.use_history_residual = use_history_residual
        self.model = self._build()

    def _build(self):
        seq_input = tf.keras.Input(
            shape=(self.sequence_length, self.input_dim), name="sequence_input"
        )
        x = tf.keras.layers.Masking(mask_value=0.0, name="pad_mask")(seq_input)
        x = tf.keras.layers.LSTM(
            self.hidden_units,
            return_sequences=False,
            dropout=self.dropout,
            name="lstm",
        )(x)
        x = tf.keras.layers.LayerNormalization(name="lstm_norm")(x)
        seq_logits = tf.keras.layers.Dense(
            self.output_dim, activation=None, name="seq_logits"
        )(x)

        if self.use_history_residual:
            hist_input = tf.keras.Input(shape=(self.output_dim,), name="history_input")
            hist_logits = RepeatGate(name="repeat_gate")(hist_input)
            logits = tf.keras.layers.Add(name="combined_logits")([seq_logits, hist_logits])
            outputs = tf.keras.layers.Activation("sigmoid", name="basket_output")(logits)
            model = tf.keras.Model([seq_input, hist_input], outputs, name="next_basket_lstm")
        else:
            outputs = tf.keras.layers.Activation("sigmoid", name="basket_output")(seq_logits)
            model = tf.keras.Model(seq_input, outputs, name="next_basket_lstm")

        loss_fn = (
            WeightedBinaryCrossentropy(pos_weight=self.pos_weight)
            if self.pos_weight != 1.0
            else tf.keras.losses.BinaryCrossentropy()
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=loss_fn,
        )
        return model