import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Go wangzh")
    parser.add_argument('--dataset', type=str, default='email-Eu', help="available datasets: "
    "[congress-bills, email-Eu, tags-math-sx, contact-primary-school, NDC-substances, email-Enron, NDC-classes, tags-ask-ubuntu, NDC-classes]")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_epoch', type=int, default=500, help='number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    # GCN layer
    parser.add_argument('--initial_dim', type=int, default=128, help='Dimensions of initial features.')
    parser.add_argument('--hidden_dim1', type=int, default=128, help='Dimensions of hidden units.')
    parser.add_argument('--output_dim', type=int, default=128, help='Dimensions of output layer.')
    # MLP layer
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='Dimensions of hidden units.')
    parser.add_argument('--mlp_output_dim', type=int, default=1, help='Dimensions of output layer.')
    parser.add_argument('--att_hidden_size', type=int, default=64, help='min epoch')
    # time window
    parser.add_argument('--log_interval', type=int, default=20, help='log interval, default: 20,[20,40,...]')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--min_loss', type=int, default=10, help='min loss value.')
    parser.add_argument('--min_epoch', type=int, default=100, help='min epoch')
    # dynamic equation
    # choosing higher-order pattern
    parser.add_argument('--tau', type=float, default=1, help='Rate of selection')
    parser.add_argument('--ratio', type=int, default=1, help='negative sampling ratio')
    args = parser.parse_args()

    if args.dataset == 'email-Eu':
        args.time_num, args.dropout, args.loss_decay, args.cluster_decay = 40, 0.3, 1, 0
        args.alpha, args.cluster_num, args.weight_decay, args.loss_temp = 0, 3, 1e-4, 0.2

    return args