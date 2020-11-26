import tensorflow as tf
import tensorflow.keras.layers as KL


def fpn_graph(input_tensor=None, 
              pyramid_size=256,
              backbone=None,
              weights=None,
              extended_layer=None):
    """Build FPN graph

    Arguments:
        input_tensor: input tensor.
        pyramid_size:
        backbone: keras model, or string
                  in string case: ResNet50V2,
        weights:
        extended_layer: None, P6, P6P7
    Returns:
        Output tensors of FPN
    """

    layer_names = [
        'conv2_block3_1_relu',
        'conv3_block4_1_relu',
        'conv4_block6_1_relu',
        'conv5_block3_1_relu',
    ]

    if isinstance(backbone, str):
        assert backbone in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']

        if backbone == 'ResNet101V2':
            layer_names[2] = 'conv4_block23_1_relu'
        elif backbone == 'ResNet152V2':
            layer_names[1] = 'conv3_block8_1_relu'
            layer_names[2] = 'conv4_block26_1_relu'

        backbone = getattr(tf.keras.applications, backbone)(
            include_top=False,
            input_tensor=input_tensor,
            pooling=None,
            weights=weights
        )

        C5 = backbone.get_layer(layer_names[3]).output
        C4 = backbone.get_layer(layer_names[2]).output
        C3 = backbone.get_layer(layer_names[1]).output
        C2 = backbone.get_layer(layer_names[0]).output

    else:
        C2, C3, C4, C5 = backbone.outputs

    P5 = KL.Conv2D(pyramid_size, 1, name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                   KL.Conv2D(pyramid_size, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                   KL.Conv2D(pyramid_size, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                   KL.Conv2D(pyramid_size, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(pyramid_size, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(pyramid_size, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(pyramid_size, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(pyramid_size, (3, 3), padding="SAME", name="fpn_p5")(P5)
    P6 = KL.Conv2D(pyramid_size, kernel_size=3, strides=2, padding="SAME", name="fpn_p6")(P5)
    P7 = KL.Conv2D(pyramid_size, kernel_size=3, strides=2, padding="SAME", name="fpn_p7")(P6)

    outputs = [P2, P3, P4, P5]
    if extended_layer:
        outputs.append(P6)
    if extended_layer == 'P6P7':
        outputs.append(P7)
    for o in outputs:
        print(o.shape)
    return outputs
    

def FPN(input_tensor=None,
        pyramid_size=256,
        backbone=None,
        weights=None,
        extended_layer=None):
    """Build FPN model

    Arguments:
        input_tensor: input tensor.
        pyramid_size:
        backbone: keras model, or string
                  in string case: ResNet50V2,
        weights:
        extended_layer: None, P6, P6P7
    Returns:
        FPN keras model
    """
    outputs = fpn_graph(input_tensor=input_tensor, 
                        pyramid_size=pyramid_size,
                        backbone=backbone,
                        weights=weights,
                        extended_layer=extended_layer)

    model = tf.keras.Model([input_tensor], outputs, name="fpn")
    return model
