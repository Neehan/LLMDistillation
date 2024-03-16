import argparse
import progressive_distillation
import full_distillation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distill", help="type of distillation", default="full", type=str
    )
    args = parser.parse_args()
    if args.distill == "progressive":
        progressive_distillation.main()
    else:
        full_distillation.main()
