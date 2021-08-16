import argparse
import json

class Options():
    """Class to manage options using parser and namespace.
    """
    def __init__(self):
        self.initialized = False

    def add_arguments_parser(self, parser: argparse.ArgumentParser):
        """Add a set of arguments to the parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to add arguments to.

        Returns
        -------
        parser : argparse.ArgumentParser
            Parser with added arguments.
        """
        # options related to datasets
        g_data = parser.add_argument_group('Data')

        # options related to model
        g_model = parser.add_argument_group('Model')
        g_model.add_argument('--name', type=str, default='pann',
                             help='Name of the model')

        # options related to training
        g_training = parser.add_argument_group('Training')
        g_training.add_argument('--load_checkpoint', type=str, default='',
                               help='Path to load a previous trained model if '
                               'not empty (default empty)')
        g_training.add_argument('--batch_size', type=int, default=32,
                                help='Batch size')
        g_training.add_argument('--learning_rate', type=float, default=1e-4,
                                help='Learning rate')
        g_training.add_argument('--optimizer', type=str, default='AMSgrad',
                                help=('Optimizer used for training, one of ',
                                      'SGD, Adam, AMSGrad (default)'))
        g_training.add_argument('--n_cycles', type=int, default=5,
                                help='Number of validation/checkpoint cycles')
        g_training.add_argument('--iters_per_cycle', type=int, default=1000,
                                help='Number of training iterations per cycle')
        g_training.add_argument('--save_freq', type=int, default=1,
                                help='Frequency (in terms of cycles) of '
                                'saving the model')
        g_training.add_argument('--training_time', type=int, default=0,
                                help=('Maximum time of training (in seconds), '
                                      'no limit if 0 (default)'))

        # options related to data augmentation
        g_training = parser.add_argument_group('Augmentation')
        g_training.add_argument('--mixup', action='store_true',
                                help='Use mix-up during training')

        self.initialized = True
        return parser

    def initialize_options(self):
        """Initialize a namespace that store options.

        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.add_arguments_parser(parser)
            self.parser = parser

        else:
            print("WARNING: Options was already initialized before")

        return self.parser.parse_args()

    def parse(self):
        """Initialize a namespace that store options.

        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        opt = self.initialize_options()
        return opt

    def print_options(self, opt: argparse.Namespace):
        """Print all options and the default values (if changed).

        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to print.
        """
        # create a new parser with default arguments
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.add_arguments_parser(parser)

        message = '----------------- Options ---------------\n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(key)
            if value != default:
                comment = f'(default {default})'
            key, value = str(key), str(value)
            message += f'{key}: {value} {comment}\n'
        print(message)

    def save_options(self, opt: argparse.Namespace, path: str):
        """Save options to a json file.

        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to save.
        path : str
            Path to save the options (.json extension will
            be automatically added at the end if absent).
        """
        if not path.endswith('.json'):
            path += '.json'
        with open(path, 'w') as f:
            f.write(json.dumps(vars(opt), indent=2))

    def load_options(self, path: str):
        """Load options from a json file.

        Parameters
        ----------
        path : str
            Path to load the options (.json extension will
            be automatically added at the end if absent).

        Returns
        -------
        opt : argparse.Namespace
            Namespace with loaded options.
        """
        if not path.endswith('.json'):
            path += '.json'
        # init a new namespace with default arguments
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opt = self.add_arguments_parser(parser).parse_args([])

        variables = json.load(open(path, 'r'))
        for key, value in variables.items():
            setattr(opt, key, value)
        return opt
