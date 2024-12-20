from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *


class EncoderHead(Layer):
    def __init__(self, n_components, use_bn=False, **kwargs):
        super(EncoderHead, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.batch_normalization = BatchNormalization()
        self.dense = Dense(n_components, activation='linear', use_bias= not use_bn)


    def call(self, inputs, training=False):
        x = self.batch_normalization(inputs, training=training) if self.use_bn else inputs
        x = self.dense(x)

        return x
    

def build_encoder(input_shape, units, n_components, activation='relu', use_bn=False):
    encoder = Sequential([
        Input(shape=input_shape),
        Dense(units, activation=activation),
        Dense(units//2, activation=activation),
        Dense(units//4, activation=activation),
        EncoderHead(n_components, use_bn)
    ], name='encoder')

    return encoder


def build_decoder(output_shape, units, n_components, activation='relu'):
    decoder = Sequential([
        Input(shape=(n_components,)),
        Dense(units//4, activation=activation),
        Dense(units//2, activation=activation),
        Dense(units, activation=activation),
        Dense(*output_shape, activation='linear')
    ], name='decoder')

    return decoder


class ConvBlock2D(Layer):
    def __init__(self, filters, dropout=0.0, **kwargs):
        super(ConvBlock2D, self).__init__(**kwargs)
        self.conv1 = Conv2D(
            filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.conv2 = Conv2D(
            filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.dropout = Dropout(dropout)
        self.maxpool = MaxPool2D(pool_size=(2, 2))


    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(inputs=x, training=training)

        return x


def build_conv_encoder(input_shape, filters, n_components, zero_padding=(0, 0), dropout=0.0, use_bn=False):
    encoder = Sequential([
        Input(shape=input_shape),
        ZeroPadding2D(zero_padding),
        ConvBlock2D(filters, dropout=dropout),
        ConvBlock2D(filters*2, dropout=dropout),
        ConvBlock2D(filters*4, dropout=dropout),
        Flatten(),
        Dense(16*n_components, activation='relu'),
        EncoderHead(n_components, use_bn)
    ], name='encoder')

    return encoder


class UpConvBlock2D(Layer):
    def __init__(self, filters, dropout=0.0, **kwargs):
        super(UpConvBlock2D, self).__init__(**kwargs)

        self.conv_transpose = Conv2DTranspose(
            filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
        self.conv1 = Conv2D(
            filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.conv2 = Conv2D(
            filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.dropout = Dropout(dropout)


    def call(self, inputs, training=False):
        x = self.conv_transpose(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(inputs=x, training=training)

        return x
    

def build_conv_decoder(output_shape, filters, n_components, cropping=(0, 0), dropout=0.0):
    # Calculate the final spatial dimensions of the encoded feature map (reverse of Flatten)
    h = (output_shape[0] + 2*cropping[0]) // 8
    w = (output_shape[1] + 2*cropping[1]) // 8
    c = filters * 4
    
    decoder = Sequential([
        Input(shape=(n_components,)),  # Input is the same size as the encoder's output (latent space)
        Dense(units=16*n_components, activation='relu'),  
        Dense(units=h * w * c, activation='relu'), # Project back to spatial dimensions
        Reshape((h, w, c)),  # Reshape back to feature map
        UpConvBlock2D(filters * 4, dropout=dropout),  # Reverse of ConvBlock(filters * 4)
        UpConvBlock2D(filters * 2, dropout=dropout),  # Reverse of ConvBlock(filters * 2)
        UpConvBlock2D(filters, dropout=dropout),      # Reverse of ConvBlock(filters)
        Conv2D(output_shape[-1], kernel_size=(1, 1), activation='sigmoid', padding='same'),  # Output layer
        Cropping2D(cropping)
    ], name='decoder')

    return decoder


class ConvBlock1D(Layer):
    def __init__(self, filters, dropout=0.0, **kwargs):
        super(ConvBlock1D, self).__init__(**kwargs)
        self.conv1 = Conv1D(
            filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.conv2 = Conv1D(
            filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.dropout = Dropout(dropout)
        self.maxpool = MaxPool1D(pool_size=2)


    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(inputs=x, training=training)

        return x


def build_seq_encoder(input_shape, filters, n_components, zero_padding=0, dropout=0.0, use_bn=False):
    encoder = Sequential([
        Input(shape=input_shape),
        ZeroPadding1D(zero_padding),
        ConvBlock1D(filters, dropout=dropout),
        ConvBlock1D(filters*2, dropout=dropout),
        ConvBlock1D(filters*4, dropout=dropout),
        Flatten(),
        EncoderHead(n_components, use_bn)
    ], name='encoder')

    return encoder


class UpConvBlock1D(Layer):
    def __init__(self, filters, dropout=0.0, **kwargs):
        super(UpConvBlock1D, self).__init__(**kwargs)

        self.conv_transpose = Conv1DTranspose(
            filters,
            kernel_size=2,
            strides=2,
            padding='same'
        )
        self.conv1 = Conv1D(
            filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.conv2 = Conv1D(
            filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_initializer = "he_normal"
        )
        self.dropout = Dropout(dropout)


    def call(self, inputs, training=False):
        x = self.conv_transpose(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(inputs=x, training=training)

        return x
    

def build_seq_decoder(output_shape, filters, n_components, cropping=0, dropout=0.0):
    l = (output_shape[0] + 2*cropping) // 8
    c = filters * 4
    
    decoder = Sequential([
        Input(shape=(n_components,)),  # Input is the same size as the encoder's output (latent space)
        Dense(units= l * c, activation='relu'),  # Project back to spatial dimensions
        Reshape((l, c)),  # Reshape back to feature map
        UpConvBlock1D(filters * 4, dropout=dropout),  # Reverse of ConvBlock(filters * 4)
        UpConvBlock1D(filters * 2, dropout=dropout),  # Reverse of ConvBlock(filters * 2)
        UpConvBlock1D(filters, dropout=dropout),      # Reverse of ConvBlock(filters)
        Conv1D(output_shape[-1], kernel_size=1, activation='linear', padding='same'),  # Output layer
        Cropping1D(cropping)
    ], name='decoder')

    return decoder