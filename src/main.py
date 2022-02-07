import time
import shutil
import argparse
import yaml
import traceback
import torch
from trainers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', nargs='?', default='recon.yaml',
            help='YAML configuration file')
    parser.add_argument('--trainer', type=str, 
            help='[Classifier]')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
            help='Resume training from last checkpoint')
    parser.add_argument('--run', type=str, default='run', 
                        help='Path for saving related data')
    parser.add_argument('--comment', type=str, default='', 
                        help='Experiment info')
    parser.add_argument('--verbose', type=str, default='info', 
                        help='[info, debug, warning, critical]')
    parser.add_argument('--test', action='store_true', 
                        help='Whether to test the model')
    # experiment parameters
    parser.add_argument('--retrain_epochs', type=int, default=20, 
                        help="rounds of retraining")

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)

    # path to save checkpoints
    args.log = os.path.join(args.run, 'logs', new_config.training.exp_id)
    ckpt_dir = os.path.join(new_config.training.save_dir, new_config.training.exp_id)
    ckpt_prefix = os.path.join(new_config.training.save_dir, "model")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    new_config.ckpt_dir = ckpt_dir
    new_config.output_dir = args.log

    clf_ckpt_dir = os.path.join(new_config.training.save_dir, new_config.training.clf_id)
    new_config.clf_ckpt_dir = clf_ckpt_dir

    if not args.test:
        if not args.resume:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)

        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        # printing messages
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # add device
    logging.info('Using device: {}'.format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    if isinstance(config, list):
        config = config[0]
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    # Initialize configuration
    args, config, logger = parse_args()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    print(config)
    print("<" * 80)

    try:
        trainer = eval(args.trainer)(config, logger)
        if not args.test:
            trainer.train()
            trainer.test()
        else:
            ckpt_pth = os.path.join(config.ckpt_dir, 'model_best.pth')
            trainer._resume_checkpoint(ckpt_pth)
            
            # test
            test_loss, test_acc = trainer.test()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
