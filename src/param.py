import argparse

parser = argparse.ArgumentParser(description='parameters')

# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
parser.add_argument('--input_image', type=str, default='../data/example.jpg',
                    help='input image file path (default: ../data/example.jpg)')
parser.add_argument('--output_folder', type=str, default='../data/',
                    help='image output folder (default: ../data/)')

config, _ = parser.parse_known_args()