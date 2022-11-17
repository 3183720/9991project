from argparse import ArgumentParser
from configs.paths_config import model_paths
from dataclasses import dataclass


@dataclass
class TrainOptionsDataClass:

    exp_dir: str
    dataset_type: str = "ffhq_encode"
    encoder_type: str = "GradualStyleEncoder"
    input_nc: int = 3
    label_nc: int = 0

    batch_size: int = 4
    test_batch_size: int = 2
    workers: int = 4
    test_workers: int = 2

    learning_rate: float = 0.0001
    optim_name: str = "ranger"
    train_decoder: bool = False
    start_from_latent_avg: bool = False

    learn_in_w: bool = False

    lpips_lambda: float = 0.8
    id_lambda: float = 0.1
    id_lambda_input: float = 0.0
    l2_lambda: float = 1.0
    latent_lambda: float = 0.0
    w_norm_lambda: float = 0
    lpips_lambda_crop: float = 0
    l2_lambda_crop: float = 0

    stylegan_weights: str = model_paths["stylegan_ffhq"]
    checkpoint_path: str = None

    max_steps: int = 500000
    image_interval: int = 100
    board_interval: int = 50
    val_interval: int = 1000
    save_interval: int = None

    # arguments for super-resolution
    resize_factors: str = None

    c_dim: int = 0


class TrainOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument(
            "--exp_dir", type=str, help="Path to experiment output directory"
        )
        self.parser.add_argument(
            "--dataset_type",
            default="ffhq_encode",
            type=str,
            help="Type of dataset/experiment to run",
        )
        self.parser.add_argument(
            "--encoder_type",
            default="GradualStyleEncoder",
            type=str,
            help="Which encoder to use",
        )
        self.parser.add_argument(
            "--input_nc",
            default=3,
            type=int,
            help="Number of input image channels to the psp encoder",
        )
        self.parser.add_argument(
            "--label_nc",
            default=0,
            type=int,
            help="Number of input label channels to the psp encoder",
        )
        self.parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
        self.parser.add_argument(
            "--batch_size", default=4, type=int, help="Batch size for training"
        )
        self.parser.add_argument(
            "--test_batch_size",
            default=2,
            type=int,
            help="Batch size for testing and inference",
        )
        self.parser.add_argument(
            "--workers",
            default=4,
            type=int,
            help="Number of train dataloader workers",
        )
        self.parser.add_argument(
            "--test_workers",
            default=2,
            type=int,
            help="Number of test/inference dataloader workers",
        )

        self.parser.add_argument(
            "--learning_rate",
            default=0.0001,
            type=float,
            help="Optimizer learning rate",
        )
        self.parser.add_argument(
            "--optim_name",
            default="ranger",
            type=str,
            help="Which optimizer to use",
        )
        self.parser.add_argument(
            "--train_decoder",
            default=False,
            type=bool,
            help="Whether to train the decoder model",
        )
        self.parser.add_argument(
            "--start_from_latent_avg",
            action="store_true",
            help="Whether to add average latent vector to generate codes from encoder.",
        )
        self.parser.add_argument(
            "--learn_in_w",
            action="store_true",
            help="Whether to learn in w space insteaf of w+",
        )

        self.parser.add_argument(
            "--c_dim",
            type=int,
            help="LPIPS loss multiplier factor",
        )

        self.parser.add_argument(
            "--lpips_lambda",
            default=0.8,
            type=float,
            help="LPIPS loss multiplier factor",
        )
        self.parser.add_argument(
            "--id_lambda",
            default=0.1,
            type=float,
            help="ID loss multiplier factor",
        )
        self.parser.add_argument(
            "--l2_lambda",
            default=1.0,
            type=float,
            help="L2 loss multiplier factor",
        )
        self.parser.add_argument(
            "--w_norm_lambda",
            default=0,
            type=float,
            help="W-norm loss multiplier factor",
        )
        self.parser.add_argument(
            "--lpips_lambda_crop",
            default=0,
            type=float,
            help="LPIPS loss multiplier factor for inner image region",
        )
        self.parser.add_argument(
            "--l2_lambda_crop",
            default=0,
            type=float,
            help="L2 loss multiplier factor for inner image region",
        )

        self.parser.add_argument(
            "--stylegan_weights",
            default=model_paths["stylegan_ffhq"],
            type=str,
            help="Path to StyleGAN model weights",
        )
        self.parser.add_argument(
            "--checkpoint_path",
            default=None,
            type=str,
            help="Path to pSp model checkpoint",
        )

        self.parser.add_argument(
            "--max_steps",
            default=500000,
            type=int,
            help="Maximum number of training steps",
        )
        self.parser.add_argument(
            "--image_interval",
            default=100,
            type=int,
            help="Interval for logging train images during training",
        )
        self.parser.add_argument(
            "--board_interval",
            default=50,
            type=int,
            help="Interval for logging metrics to tensorboard",
        )
        self.parser.add_argument(
            "--val_interval",
            default=1000,
            type=int,
            help="Validation interval",
        )
        self.parser.add_argument(
            "--save_interval",
            default=None,
            type=int,
            help="Model checkpoint interval",
        )

        # arguments for super-resolution
        self.parser.add_argument(
            "--resize_factors",
            type=str,
            default=None,
            help="For super-res, comma-separated resize factors to use for inference.",
        )

    def parse(self):
        opts = self.parser.parse_args()
        return opts
