#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import sys
import json
import common


def myParser():
    parser = ArgumentParser(description="Train a Full Cost-Volume Network.")

    parser.add_argument(
        "--experiment_root",
        default="FL_Aug_MEL",
        type=common.float_or_string,
        help="Location used to store checkpoints and dumped data.",
    )

    parser.add_argument(
        "--num_classes",
        default=7,
        type=int,
        help="Number of classes in the output of the network.",
    )

    parser.add_argument(
        "--skip_class",
        default="MEL",
        type=common.float_or_string,
        help="New class label to skip in training.",
    )

    parser.add_argument(
        "--dataset_root",
        default="/usr/xtmp/hannah/segkeras/nn-isic2019/dataset/",
        type=common.float_or_string,
        help="Path to the dataset of images for training and testing.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="When this flag is provided, all other arguments apart from the "
        "experiment_root are ignored and a previously saved set of arguments "
        "is loaded.",
    )

    parser.add_argument(
        "--initial_checkpoint",
        default=None,
        help="Path to the checkpoint file of the pretrained network.",
    )

    parser.add_argument(
        "--batch_size",
        default=40,
        type=common.positive_int,
        help="The number of samples in a batch",
    )

    parser.add_argument(
        "--is_train",
        default=1,
        type=int,
        help="The train/val/test split of the dataset to use: 1 - training, 0 - testing, 2 - validation.",
    )

    parser.add_argument(
        "--loading_threads",
        default=8,
        type=common.positive_int,
        help="Number of threads used for parallel loading.",
    )

    parser.add_argument(
        "--margin",
        default="soft",
        type=common.float_or_string,
        help='What margin to use: a float value for hard-margin, "soft" for '
        'soft-margin, or no margin if "none".',
    )

    parser.add_argument(
        "--learning_rate",
        default=0.000001,
        type=common.positive_float,
        help="The initial value of the learning-rate, before it kicks in.",
    )

    parser.add_argument(
        "--train_iterations",
        default=25000,
        type=common.positive_int,
        help="Number of training iterations.",
    )

    parser.add_argument(
        "--decay_start_iteration",
        default=15000,
        type=int,
        help="At which iteration the learning-rate decay should kick-in."
        "Set to -1 to disable decay completely.",
    )

    parser.add_argument(
        "--checkpoint_frequency",
        default=1000,
        type=common.nonnegative_int,
        help="After how many iterations a checkpoint is stored. Set this to 0 to "
        "disable intermediate storing. This will result in only one final "
        "checkpoint.",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="When this flag is provided, data augmentation is performed.",
    )

    parser.add_argument(
        "--detailed_logs",
        action="store_true",
        default=False,
        help="Store very detailed logs of the training in addition to TensorBoard"
        " summaries. These are mem-mapped numpy files containing the"
        " embeddings, losses and FIDs seen in each batch during training."
        " Everything can be re-constructed and analyzed that way.",
    )

    parser.add_argument("--gpu_device", default="0", help="GPU device ID.")

    parser.add_argument(
        "--num_gpus", default="1", type=int, help="Number of GPU to use."
    )

    parser.add_argument(
        "--heavy_augment", action="store_true", default=False, help="GPU device ID."
    )

    parser.add_argument(
        "--load_weights",
        type=common.float_or_string,
        help="Path to load previously trained weights from experiment folder. If given an empty string, dont load any weights.",
    )

    parser.add_argument(
        "--init_epoch",
        default=0,
        type=common.nonnegative_int,
        help="The initial epoch # to start training",
    )

    parser.add_argument(
        "--num_epochs",
        default=25,
        type=common.positive_int,
        help="Total number of epochs for training",
    )

    parser.add_argument(
        "--early_stop_patience",
        default=8,
        type=common.positive_int,
        help="Total number of epochs for training",
    )

    parser.add_argument(
        "--loss_type", default="1", type=int, help="Type of Loss Function Used."
    )

    parser.add_argument(
        "--sizeH", default="224", type=int, help="Height of the input image."
    )

    parser.add_argument(
        "--sizeW", default="224", type=int, help="Width of the input image."
    )

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # We store all arguments in a json file. This has two advantages:
    #    1. We can always get back and see what exactly that experiment was
    #    2. We can resume an experiment as-is without needing to remember all flags.
    args_file = os.path.join("experiments/", args.experiment_root, "args.json")
    if args.resume:
        if not os.path.isfile(args_file):
            raise IOError("`args.json` not found in {}".format(args_file))

        print("Loading args from {}.".format(args_file))
        with open(args_file, "r") as f:
            args_resumed = json.load(f)
        args_resumed["resume"] = True  # This would be overwritten.

        # When resuming, we not only want to populate the args object with the
        #  values from the file, but we also want to check for some possible
        #  conflicts between loaded and given arguments.
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if resumed_value != value and key != "init_epoch":
                    print(
                        "Warning: For the argument `{}` we are using the"
                        " loaded value `{}`. The provided value was `{}`"
                        ".".format(key, resumed_value, value)
                    )
                    args.__dict__[key] = resumed_value
            else:
                print(
                    "Warning: A new argument was added since the last run:"
                    " `{}`. Using the new value: `{}`.".format(key, value)
                )
    elif args.is_train != 1:
        if not os.path.isfile(args_file):
            raise IOError("`args.json` not found in {}".format(args_file))

        print("Loading args from {}.".format(args_file))
        with open(args_file, "r") as f:
            args_resumed = json.load(f)
        args_resumed["resume"] = True  # This would be overwritten.

        # When resuming, we not only want to populate the args object with the
        #  values from the file, but we also want to check for some possible
        #  conflicts between loaded and given arguments.
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if (
                    resumed_value != value
                    and key != "is_train"
                    and key != "batch_size"
                    and key != "num_gpus"
                    and key != "gpu_device"
                    and key != "load_weights"
                ):
                    print(
                        "Warning: For the argument `{}` we are using the"
                        " loaded value `{}`. The provided value was `{}`"
                        ".".format(key, resumed_value, value)
                    )
                    args.__dict__[key] = resumed_value
            else:
                print(
                    "Warning: A new argument was added since the last run:"
                    " `{}`. Using the new value: `{}`.".format(key, value)
                )
    elif args.is_train == 1:
        # If the experiment directory exists already, we bail in fear.
        if os.path.exists("experiments/" + args.experiment_root):
            if (
                os.listdir("experiments/" + args.experiment_root)
                and args.init_epoch != 0
            ):
                print(
                    "The directory {} already exists and is not empty. If "
                    "you want to resume training, append --resume to your "
                    "call.".format("experiments/" + args.experiment_root)
                )
        #                exit(1)
        else:
            os.makedirs("experiments/" + args.experiment_root)

        # Store the passed arguments for later resuming and grepping in a nice
        # and readable format.
        with open(args_file, "w") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    # Also show all parameter values at the start, for ease of reading logs.
    print("Training using the following parameters:")
    for key, value in sorted(vars(args).items()):
        print("{}: {}".format(key, value))

    # Check them here, so they are not required when --resume-ing.
    if not args.dataset_root:
        parser.print_help()
        print("You did not specify the required `dataset_root` argument!")
        sys.exit(1)

    return args
