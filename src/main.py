import argparse
import progressive_distillation
import full_distillation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distill", help="type of distillation", default="full", type=str
    )
    parser.add_argument(
        "--train-attn",
        dest="train_attn",
        action="store_true",
        help="train the attention layer",
    )
    parser.set_defaults(train_attn=False)

    args = parser.parse_args()
    if args.distill == "progressive":
        progressive_distillation.main(args.train_attn)
    else:
        full_distillation.main(args.train_attn)
