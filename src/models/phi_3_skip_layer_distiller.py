import logging
from src.skip_layer_distiller import SkipLayerDistiller, SkippableLayer
from src.training_loop import training_loop
from src.argparser import parser


class PhiMatDistiller(SkipLayerDistiller):
    def get_model_layers(self, model):
        return model.model.layers

    def augment_student_model(self, layer_id):
        self.student_model.model.layers[layer_id] = SkippableLayer(
            self.student_model.model.layers[layer_id],
            self.active_layer_list,
            layer_id,
        )

        # Disable gradients for all layers except the MLP layer just replaced
        for name, param in self.student_model.named_parameters():
            if f"model.layers.{layer_id}" not in name:
                param.requires_grad = False


if __name__ == "__main__":

    parser.add_argument(
        "--model",
        help="the model to distill",
        default="microsoft/Phi-3-mini-128k-instruct",
        type=str,
    )

    # parser.add_argument(
    #     "--num_layers",
    #     help="the number of layers in the model",
    #     default=32,
    #     type=int,
    # )

    args = parser.parse_args()
    for arg, value in vars(args).items():
        logging.info(f"Argument: {arg}, Value: {value}")

    distiller_kwargs = {}
    training_loop(distiller_factory=PhiMatDistiller, args=args)
