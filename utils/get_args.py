import argparse


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--hash-layer", type=str, default="select", help="choice a hash layer [select, linear] to run. select: select mechaism, linear: sign function.")
    parser.add_argument("--save-dir", type=str, default="./resultF/64-bit")
    parser.add_argument("--clip-path", type=str, default="./ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--dataset", type=str, default="coco", help="choise from [coco, mirflckr25k, nuswide]")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")
    parser.add_argument("--similarity-function", type=str, default="euclidean", help="choise form [cosine, euclidean]")
    parser.add_argument("--loss-type", type=str, default="l2", help="choise form [l1, l2]")
    # parser.add_argument("--test-index-file", type=str, default="./data/test/index.mat")
    # parser.add_argument("--test-caption-file", type=str, default="./data/test/captions.mat")
    # parser.add_argument("--test-label-file", type=str, default="./data/test/label.mat")

    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-num", type=int, default=5120)
    parser.add_argument("--train-num", type=int, default=10240)
    parser.add_argument("--lr-decay-freq", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1814)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--clip-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--vartheta", type=float, default=0.5, help="the rate of error code.")
    parser.add_argument("--sim-threshold", type=float, default=0.1)

    parser.add_argument("--is-train", action="store_true")
    ####################################################
    parser.add_argument("--n-classes", type=int, default=24, help="the number of dataset classes")
    parser.add_argument("--n-bits", type=int, default=128, help="length of hashing binary")
    parser.add_argument("--topk", type=int, default=5000, help="mAP@topk")

    parser.add_argument("--lr-center", default=0.01, type=float, help="initial center learning rate")
    parser.add_argument("--freeze-BN", default=True, type=bool, help="freeze bn")
    parser.add_argument("--scale", default=23.0, type=float, help="scale for softmax variations")
    parser.add_argument("--ps-mu", default=1.0, type=float, help="generation ratio in proxy synthesis")
    parser.add_argument("--ps-alpha", default=0.4, type=float, help="alpha for beta distribution in proxy synthesis")
    parser.add_argument("--ps-rate", default=0.4, type=float, help="alpha for beta distribution in proxy synthesis")
    parser.add_argument("--method", type=int, default=2, help="loss func #")
    ####################################################
    parser.add_argument(
        "--sigmoid_temperature",
        default=0.01,
        type=float,
        help="SmoothAP: the temperature of the sigmoid used in SmoothAP loss",
    )

    args = parser.parse_args()

    return args

