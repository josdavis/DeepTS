#from builtins import object
from keras.models import Model
from keras.layers import Input
from importlib import import_module

class KerasModelFactory(object):
    def __init__(self):
        self.keras_layers_path = "keras.layers"

    def create_keras_model(self, nn_architecture_dict):
        """

        :param nn_architecture_dict: A dictionary that contains the network
        architecture and the different arguments for each layer.
        An example can be found at lstm_model.yaml
        :return: The compiled Keras model.
        """
        nodes = {}
        ips = nn_architecture_dict["inputs"]
        existing_nodes = []

        #  Create input layer nodes and add them to the node directory
        for k in ips.keys():
            temp_dict = ips[k]
            temp_input = Input(shape=temp_dict["shape"],
                               name=temp_dict["name"])
            nodes[temp_dict["name"]] = temp_input
            existing_nodes.append(temp_dict["name"])

        layers_dict = nn_architecture_dict["layers"]
        num_of_layers = len(layers_dict.keys())

        #  This assumes that that the numbering in the architecture dictionary
        #  is ascending. Any layer which has a number of i cannot have layers
        #  with number >i as its input.
        for i in range(num_of_layers):
            layer_info = layers_dict[i]
            module = import_module(self.keras_layers_path)
            theClass = getattr(module,layer_info["type"])
            if layer_info["type"] == "concatenate":
                assert set(layer_info["input"]).issubset(set(nodes.keys())),\
                    "Ensure that each layer built is in order. Trying to " \
                    "build a layer whose input in not built yet will throw " \
                    "this assertion error"
                output_node = theClass([nodes[x] for x in layer_info["input"]],
                                       **layer_info["type_arguments"])
                nodes[layer_info["output_name"]] = output_node
            else:
                assert layer_info["input"] in nodes.keys(), \
                    "Ensure that each layer built is in order. Trying to " \
                    "build a layer whose input in not built yet will throw " \
                    "this assertion error"
                output_node = theClass(**layer_info["type_arguments"])(
                    nodes[layer_info["input"]]
                )
                nodes[layer_info["output_name"]] = output_node

        model_config = nn_architecture_dict["model"]
        model = Model(inputs=[nodes[x] for x in model_config["inputs"]],
                      outputs=[nodes[x] for x in model_config["outputs"]])
        model.compile(**model_config["compile_arguments"])
        return model
