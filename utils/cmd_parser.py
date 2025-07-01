import argparse
import sys
from ruamel.yaml import YAML

def get_agrs_parser():
    parser = argparse.ArgumentParser(
        description="Parser Example",
        add_help=True
    )
    parser.add_argument('-c', '--config_file', help='Specify config file', metavar='FILE')
    subparser = parser.add_subparsers(dest='mode')
    parser_train = subparser.add_parser('train')
    parser_eval = subparser.add_parser('test')
    parser_train.add_argument('--seed', type=int, default=0, metavar='N')
    parser_train.add_argument('--num_workers', type=int, default=4, metavar='N')
    parser_train.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
    parser_eval.add_argument('--num_workers', type=int, default=4, metavar='N')
    parser_eval.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')

    args = parser.parse_args() # get all arguments in the parser
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        print(f'{key:25s} -> {value}')

    yaml = YAML()
    yaml.indent(mapping = 2, sequence=2, offset = 2)
    yaml.default_flow_style = False
    with open(args.config_file, 'r') as f:
        cfg = yaml.load(f)
    yaml.dump(cfg, sys.stdout, transform=replace_indent)

    return cfg, args

def replace_indent(stream):
    stream = "     " + stream
    return stream.replace("\n", "\n     ")



