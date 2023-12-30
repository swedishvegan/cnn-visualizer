import argparse

layers = ["conv1", "maxpool1", "conv2", "conv3", "maxpool2", "inception3a", "inception3b", "maxpool3", "inception4a", "inception4b", "inception4c", "inception4d", "inception4e", "maxpool4", "inception5a", "inception5b", "avgpool"]
num_features = [64, 64, 192, 192, 192, 256, 480, 480, 512, 512, 512, 528, 832, 832, 832, 1024, 1024]

assert len(layers) == len(num_features)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    default = None,
    type = str,
    required = False,
    metavar = "",
    help = "filepath of image to convert (program creates new image from scratch if no filepath is supplied"
)
parser.add_argument(
    "--size",
    default = 256,
    type = int,
    required = False,
    metavar = "",
    help = "default = 256, size of image to generate if no filepath is supplied"
)
parser.add_argument(
    "--layer",
    default = "inception4a",
    type = str,
    required = False,
    metavar = "",
    help = "default = 'inception4a', layer of network to activate",
    choices = layers
)
parser.add_argument(
    "--feature",
    default = 225,
    type = int,
    required = False,
    metavar = "",
    help = "default = 225, feature of relevant layer to focus on"
)
parser.add_argument(
    "--ss",
    default = 1.0,
    type = float,
    required = False,
    metavar = "",
    help = "default = 1, step size in the image optimization loop"
)
parser.add_argument(
    "--magnify",
    default = 1,
    type = int,
    required = False,
    choices = range(2, 6),
    metavar = "",
    help = "default = 1, max = 5, magnifies the scale of patterns generated"
)
parser.add_argument(
    "--colorfulness",
    default = 0.0,
    type = float,
    required = False,
    metavar = "",
    help = "default = 0, affects penalty for extreme pixel values"
)
parser.add_argument(
    "--sharpness",
    default = 0.0,
    type = float,
    required = False,
    metavar = "",
    help = "default = 0, affects penalty for sharp color gradients"
)
parser.add_argument(
    "--intensity",
    default = 0.0,
    type = float,
    required = False,
    metavar = "",
    help = "default = 0, affects reward for activating relevant kernels"
)
parser.add_argument(
    "--octaves",
    default = 1,
    type = int,
    required = False,
    metavar = "",
    help = "default = 1, number of upscalings"
)
parser.add_argument(
    "--scaling",
    default = 2.0,
    type = float,
    required = False,
    metavar = "",
    help = "default = 2, image upscaling factor"
)
parser.add_argument(
    "--outscale",
    default = 1.0,
    type = float,
    required = False,
    metavar = "",
    help = "default = 1, output image scaling factor"
)
args = parser.parse_args()

error = ""
max_features = num_features[layers.index(args.layer)]
if args.feature >= max_features: error = "Layer "" + args.layer + "" only has " + str(max_features) + " features"
elif args.ss <= 0.0: error = "Step size must be positive"
elif args.size < 32 or args.size > 2048: error = "Size must be in range [32, 2048]"
elif args.octaves < 1 or args.octaves > 8: error = "Octaves must be in range [1, 8]"
elif args.scaling < 1.0 or args.scaling > 3.0: error = "Scaling must be in range [1, 3]"
elif args.outscale <= 0.0: error = "Output scale must be positive"

if error != "":

    print(error)
    print()
    parser.print_help()
    exit()

