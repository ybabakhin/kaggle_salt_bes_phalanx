from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, \
    Activation, Add, Input, Softmax
from keras.models import Model
from keras.backend import int_shape, is_keras_tensor
from models.conv2d_transpose import Conv2DTranspose


class LinkNet():
    """LinkNet architecture.

    The model follows the architecture presented in: https://arxiv.org/abs/1707.03718

    Args:
        num_classes (int): the number of classes to segment.
        input_tensor (tensor, optional): Keras tensor
            (i.e. output of `layers.Input()`) to use as image input for
            the model. Default: None.
        input_shape (tuple, optional): Shape tuple of the model input.
            Default: None.
        initial_block_filters (int, optional): The number of filters after
            the initial block (see the paper for details on the initial
            block). Default: None.
        bias (bool, optional): If ``True``, adds a learnable bias.
            Default: ``False``.

    """

    def __init__(
        self,
        num_classes,
        input_tensor=None,
        input_shape=None,
        initial_block_filters=64,
        bias=False,
        name='linknet'
    ):
        self.num_classes = num_classes
        self.initial_block_filters = initial_block_filters
        self.bias = bias
        self.output_shape = input_shape[:-1] + (num_classes, )

        # Create a Keras tensor from the input_shape/input_tensor
        if input_tensor is None:
            self.input = Input(shape=input_shape, name='input_img')
        elif is_keras_tensor(input_tensor):
            self.input = input_tensor
        else:
            # input_tensor is a tensor but not one from Keras
            self.input = Input(
                tensor=input_tensor, shape=input_shape, name='input_img'
            )

        self.name = name

    def get_model(
        self,
        pretrained_encoder=True,
        weights_path='./checkpoints/linknet_encoder_weights.h5'
    ):
        """Initializes a LinkNet model.

        Returns:
            A Keras model instance.

        """
        # Build encoder
        encoder_model = self.get_encoder()
        if pretrained_encoder:
            encoder_model.load_weights(weights_path)
        encoder_out = encoder_model(self.input)

        # Build decoder
        decoder_model = self.get_decoder(encoder_out)
        decoder_out = decoder_model(encoder_out[:-1])

        return Model(inputs=self.input, outputs=decoder_out, name=self.name)

    def get_encoder(self, name='encoder'):
        """Builds the encoder of a LinkNet architecture.

        Args:
            name (string, optional): The encoder model name.
                Default: 'encoder'.

        Returns:
            The encoder as a Keras model instance.

        """
        # Initial block
        initial1 = Conv2D(
            self.initial_block_filters,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/conv2d_1'
        )(self.input)
        initial1 = BatchNormalization(name=name + '/0/bn_1')(initial1)
        initial1 = Activation('relu', name=name + '/0/relu_1')(initial1)
        initial2 = MaxPooling2D(pool_size=2, name=name + '/0/maxpool_1')(initial1)  # yapf: disable

        # Encoder blocks
        encoder1 = self.encoder_block(
            initial2,
            self.initial_block_filters,
            strides=1,
            bias=self.bias,
            name=name + '/1'
        )
        encoder2 = self.encoder_block(
            encoder1,
            self.initial_block_filters * 2,
            strides=(2, 1),
            bias=self.bias,
            name=name + '/2'
        )
        encoder3 = self.encoder_block(
            encoder2,
            self.initial_block_filters * 4,
            strides=(2, 1),
            bias=self.bias,
            name=name + '/3'
        )
        encoder4 = self.encoder_block(
            encoder3,
            self.initial_block_filters * 8,
            strides=(2, 1),
            bias=self.bias,
            name=name + '/4'
        )

        return Model(
            inputs=self.input,
            outputs=[
                encoder4, encoder3, encoder2, encoder1, initial2, initial1
            ],
            name=name
        )

    def encoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        bias=False,
        name=''
    ):
        """Creates an encoder block.

        The encoder block is a combination of two basic encoder blocks
        (see ``encoder_basic_block``). The first with stride 2 and the
        the second with stride 1.

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, or list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, or list, optional): A tuple/list of two
                integers, specifying the stride for each basic block. A
                single integer can also be specified, in which case both
                basic blocks use the same stride. Default: 1.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.
            name (string, optional): A string to identify this block.
                Default: Empty string.

        Returns:
            The output tensor of the block.

        """
        assert isinstance(strides, (int, tuple, list)), (
            "expected int, tuple, or list for strides"
        )  # yapf: disable
        if (isinstance(strides, (tuple, list))):
            if len(strides) == 2:
                stride_1, stride_2 = strides
            else:
                raise ValueError("expected a list or tuple on length 2")
        else:
            stride_1 = strides
            stride_2 = strides

        x = self.encoder_basic_block(
            input,
            out_filters,
            kernel_size=kernel_size,
            strides=stride_1,
            padding=padding,
            bias=bias,
            name=name + '/1'
        )

        x = self.encoder_basic_block(
            x,
            out_filters,
            kernel_size=kernel_size,
            strides=stride_2,
            padding=padding,
            bias=bias,
            name=name + '/2'
        )

        return x

    def encoder_basic_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        bias=False,
        name=''
    ):
        """Creates a basic encoder block.

        Main brach architecture:
        1. Conv2D
        2. BatchNormalization
        3. ReLU
        4. Conv2D
        5. BatchNormalization
        Residual branch architecture:
        1. Conv2D, if `strides` is greater than 1
        The output of the main and residual branches are then added together
        with ReLU activation.

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the strides along the height and width
                of the 2D input. In case it's a single integer, it's value
                is used for all spatial dimensions. Default: 1.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.
            name (string, optional): A string to identify this block.
                Default: Empty string.

        Returns:
            The output tensor of the block.

        """
        residual = input

        x = Conv2D(
            out_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=bias,
            name=name + '/main/conv2d_1'
        )(input)
        x = BatchNormalization(name=name + '/main/bn_1')(x)
        x = Activation('relu', name=name + '/main/relu_1')(x)

        x = Conv2D(
            out_filters,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/main/conv2d_2'
        )(x)
        x = BatchNormalization(name=name + '/main/bn_2')(x)

        if strides > 1:
            residual = Conv2D(
                out_filters,
                kernel_size=1,
                strides=strides,
                padding=padding,
                use_bias=bias,
                name=name + '/res/conv2d_1'
            )(residual)
            residual = BatchNormalization(name=name + '/res/bn_1')(residual)

        x = Add(name=name + '/add')([x, residual])
        x = Activation('relu', name=name + '/relu_1')(x)

        return x

    def get_decoder(self, inputs, name='decoder'):
        """Builds the decoder of a LinkNet architecture.

        Args:
            name (string, optional): The encoder model name.
                Default: 'decoder'.

        Returns:
            The decoder as a Keras model instance.

        """
        # Decoder inputs
        encoder4 = Input(shape=int_shape(inputs[0])[1:], name='encoder4')
        encoder3 = Input(shape=int_shape(inputs[1])[1:], name='encoder3')
        encoder2 = Input(shape=int_shape(inputs[2])[1:], name='encoder2')
        encoder1 = Input(shape=int_shape(inputs[3])[1:], name='encoder1')
        initial2 = Input(shape=int_shape(inputs[4])[1:], name='initial2')
        initial1 = inputs[5]

        # Decoder blocks
        decoder4 = self.decoder_block(
            encoder4,
            self.initial_block_filters * 4,
            strides=2,
            output_shape=int_shape(encoder3)[1:],
            bias=self.bias,
            name=name + '/4'
        )
        decoder4 = Add(name=name + '/shortcut_e3_d4')([encoder3, decoder4])

        decoder3 = self.decoder_block(
            decoder4,
            self.initial_block_filters * 2,
            strides=2,
            output_shape=int_shape(encoder2)[1:],
            bias=self.bias,
            name=name + '/3'
        )
        decoder3 = Add(name=name + '/shortcut_e2_d3')([encoder2, decoder3])

        decoder2 = self.decoder_block(
            decoder3,
            self.initial_block_filters,
            strides=2,
            output_shape=int_shape(encoder1)[1:],
            bias=self.bias,
            name=name + '/2'
        )
        decoder2 = Add(name=name + '/shortcut_e1_d2')([encoder1, decoder2])

        decoder1 = self.decoder_block(
            decoder2,
            self.initial_block_filters,
            strides=1,
            output_shape=int_shape(initial2)[1:],
            bias=self.bias,
            name=name + '/1'
        )
        decoder1 = Add(name=name + '/shortcut_init_d1')([initial2, decoder1])

        # Final block
        # Build the output shape of the next layer - same width and height
        # as initial1
        shape = (
            int_shape(initial1)[1],
            int_shape(initial1)[2],
            self.initial_block_filters // 2,
        )
        final = Conv2DTranspose(
            self.initial_block_filters // 2,
            kernel_size=3,
            strides=2,
            padding='same',
            output_shape=shape,
            use_bias=self.bias,
            name=name + '/0/transposed2d_1'
        )(decoder1)
        final = BatchNormalization(name=name + '/0/bn_1')(final)
        final = Activation('relu', name=name + '/0/relu_1')(final)

        final = Conv2D(
            self.initial_block_filters // 2,
            kernel_size=3,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/conv2d_1'
        )(final)
        final = BatchNormalization(name=name + '/0/bn_2')(final)
        final = Activation('relu', name=name + '/0/relu_2')(final)

        logits = Conv2DTranspose(
            self.num_classes,
            kernel_size=2,
            strides=2,
            padding='same',
            output_shape=self.output_shape,
            use_bias=self.bias,
            name=name + '/0/transposed2d_2'
        )(final)

        prediction = Softmax(name=name + '/0/softmax')(logits)

        return Model(
            inputs=[
                encoder4, encoder3, encoder2, encoder1, initial2
            ],
            outputs=prediction,
            name=name
        )

    def decoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=2,
        projection_ratio=4,
        padding='same',
        output_shape=None,
        bias=False,
        name=''
    ):
        """Creates a decoder block.

        Decoder block architecture:
        1. Conv2D
        2. BatchNormalization
        3. ReLU
        4. Conv2DTranspose
        5. BatchNormalization
        6. ReLU
        7. Conv2D
        8. BatchNormalization
        9. ReLU

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the strides along the height and width
                of the 2D input. In case it's a single integer, it's value
                is used for all spatial dimensions. Default: 1.
            projection_ratio (int, optional): A scale factor applied to
                the number of input channels. The output of the first
                convolution will have ``input_channels // projection_ratio``.
                The goal is to decrease the number of parameters in the
                transposed convolution layer. Default: 4.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            output_shape: A tuple of integers specifying the shape of the output
                without the batch size. Default: None.
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.
            name (string, optional): A string to identify this block.
                Default: Empty string.

        Returns:
            The output tensor of the block.

        """
        internal_filters = int_shape(input)[-1] // projection_ratio
        x = Conv2D(
            internal_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/conv2d_1'
        )(input)
        x = BatchNormalization(name=name + '/bn_1')(x)
        x = Activation('relu', name=name + '/relu_1')(x)

        # The shape of the following trasposed convolution is the output
        # shape of the block with 'internal_filters' channels
        shape = output_shape[:-1] + (internal_filters, )
        x = Conv2DTranspose(
            internal_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_shape=shape,
            use_bias=bias,
            name=name + '/transposed2d_1'
        )(x)
        x = BatchNormalization(name=name + '/bn_2')(x)
        x = Activation('relu', name=name + '/relu_2')(x)

        x = Conv2D(
            out_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/conv2d_2'
        )(x)
        x = BatchNormalization(name=name + '/bn_3')(x)
        x = Activation('relu', name=name + '/relu_3')(x)

        return x