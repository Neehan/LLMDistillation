import argparse
import progressive_distillation
import full_distillation
import matryoshka_distillation
import full_matryoshka_distillation
import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distill", help="type of distillation", default="mat", type=str
    )
    parser.add_argument(
        "--model",
        help="the model to distill",
        default="pythia-160",
        type=str,
    )
    parser.add_argument(
        "--train-attn",
        dest="train_attn",
        action="store_true",
        help="train the attention layer",
    )
    parser.set_defaults(train_attn=False)

    args = parser.parse_args()
    model_path = constants.MODEL_PATHS[args.model]

    if args.distill == "progressive":
        progressive_distillation.main(model_path, args.train_attn)
    elif args.distill == "mat":
        matryoshka_distillation.main(model_path, args.train_attn)
    elif args.distill == "fullmat":
        full_matryoshka_distillation.main(model_path, args.train_attn)
    else:
        full_distillation.main(model_path, args.train_attn)
