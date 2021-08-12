import argparse
import os
import sys

import pytest

from lib.options import Options

@pytest.fixture(scope="module")
def parser():
    return

# Calling pytest <this file> --options causes the parser to get <this file>
# and options as unexpected argument(s). The following line solves the problem.
sys.argv = ['pytest']

def test_init():
    opt = Options()
    assert opt.initialized is False


def test_add_arguments_parser():
    options = Options()
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
                )
    newparser = options.add_arguments_parser(parser)
    variables = vars(newparser.parse_args())
    assert 'batch_size' in variables
    assert type(variables['batch_size']) is int
    assert 'mixup' in variables
    assert variables['mixup'] is False
    assert options.initialized is True
    group_names = [group.title for group in newparser._action_groups]
    for group_name in ['Data', 'Model', 'Training']:
        assert group_name in group_names, (f'{group_name} is missing in '
                                           'group names')


def test_initialize_options(capsys):
    options = Options()
    opt = options.initialize_options()
    assert hasattr(opt, "batch_size")
    assert hasattr(opt, "mixup")
    assert type(opt.batch_size) is int
    assert options.initialized is True
    assert options.parser.parse_args() == opt
    # test warning
    options.initialize_options()
    captured = capsys.readouterr()
    assert captured.out.startswith("WARNING: Options was already initialized")


def test_parse():
    options = Options()
    opt = options.parse()
    assert hasattr(opt, "batch_size")
    assert hasattr(opt, "mixup")
    assert type(opt.batch_size) is int
    assert options.initialized is True
    assert options.parser.parse_args() == opt


def test_print_options(capsys):
    options = Options()
    opt = options.parse()
    options.print_options(opt)
    captured = capsys.readouterr()
    assert "batch_size" in captured.out
    assert "mixup" in captured.out
    # case when a value is modified
    options = Options()
    opt = options.parse()
    opt.mixup = True
    options.print_options(opt)
    captured = capsys.readouterr()
    assert "mixup: True (default False)" in captured.out

def test_save_options():
    options = Options()
    opt = options.parse()
    # test if path ending with .json works
    options.save_options(opt, 'tests/tmp_opt.json')
    assert os.path.isfile('tests/tmp_opt.json')
    os.remove('tests/tmp_opt.json')
    # test if path that not ending with .json works
    options.save_options(opt, 'tests/tmp_opt')
    assert os.path.isfile('tests/tmp_opt.json')
    os.remove('tests/tmp_opt.json')

def test_load_options():
    options = Options()
    opt = options.parse()
    opt.mixup = True
    options.save_options(opt, 'tests/tmp_opt.json')
    # test if path ending with .json works
    opt2 = options.load_options('tests/tmp_opt.json')
    assert hasattr(opt2, 'batch_size')
    assert hasattr(opt2, 'mixup')
    assert opt.batch_size == opt2.batch_size
    assert opt2.mixup is True
    # test if path that not ending with .json works
    opt3 = options.load_options('tests/tmp_opt')
    assert hasattr(opt3, 'batch_size')
    assert hasattr(opt3, 'mixup')
    assert opt.batch_size == opt3.batch_size
    assert opt3.mixup is True

    os.remove('tests/tmp_opt.json')
