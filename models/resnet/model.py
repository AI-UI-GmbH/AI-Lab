import tensorflow as tf


def resnet_graph(input_tensor, name, weights):
    layer_names = [
        'conv2_block3_1_relu',
        'conv3_block4_1_relu',
        'conv4_block6_1_relu',
        'conv5_block3_1_relu',
    ]

    assert name in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']

    if name == 'ResNet101V2':
        layer_names[2] = 'conv4_block23_1_relu'
    elif name == 'ResNet152V2':
        layer_names[1] = 'conv3_block8_1_relu'
        layer_names[2] = 'conv4_block26_1_relu'

    backbone = getattr(tf.keras.applications, name)(
        include_top=False,
        input_tensor=input_tensor,
        pooling=None,
        weights=weights
    )

    C5 = backbone.get_layer(layer_names[3]).output
    C4 = backbone.get_layer(layer_names[2]).output
    C3 = backbone.get_layer(layer_names[1]).output
    C2 = backbone.get_layer(layer_names[0]).output

    return C2, C3, C4, C5
