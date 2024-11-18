from __future__ import division
from __future__ import print_function
from config import parse_args
from sympy import *
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import *
from models import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings(action='ignore')


def train():
    model.train()
    patience = 0
    for epoch in range(1, args.max_epoch + 1):
        epoch_losses = []
        model.train()
        for t in range(args.time_num):
            optimizer.zero_grad()
            GCN_output1, loss_lower = model.forward_lower(adj_train[t].to(args.device), pos_train[t], neg_train[t])
            GCN_output2 = model.forward_higher(high_adj[t].to(args.device))
            output11, output12 = model.dynamics(GCN_output1, degree[t].to(args.device), lap_p[t].to(args.device), lap_n[t].to(args.device))
            h1 = torch.stack([output11, output12], dim=1)
            output1, att1 = model.attention(h1.to(torch.float32))
            output2 = GCN_output1 + (1/4) * output1
            output = torch.cat((output2, GCN_output2), 1)
            loss_cat = model.compute_outside_prob(output, pos_train[t], neg_train[t])
            loss_hl = model.calculate_hl_loss(GCN_output1, GCN_output2)
            epoch_loss = loss_cat + loss_lower + 0.1 * loss_hl
            epoch_loss.backward()
            optimizer.step()
            epoch_losses.append(epoch_loss.item())
        model.eval()
        average_epoch_loss = np.mean(epoch_losses)
        if epoch == args.max_epoch:
            test_results = eval_high(int(test_window_num), pos_test, neg_test, output)
            print('==================================================================')
            print('=========================final_epoch==============================')
            print("Epoch:{}, Test AUC: {:.4f}, Test_AP: {:.4f}".format(epoch, test_results[0], test_results[1]))
            break
        elif average_epoch_loss < args.min_loss:
            args.min_loss = average_epoch_loss
            patience = 0
        else:
            patience += 1
            if epoch > args.min_epoch and patience > args.patience:
                model.eval()
                auc, ap = eval_high(int(test_window_num), pos_test, neg_test, output)
                print('==================================================================')
                print('======================early stopping==============================')
                print("Epoch:{}, Test AUC: {:.4f}, Test_AP: {:.4f}".format(epoch, auc, ap))
                break
        if epoch == 1 or epoch % args.log_interval == 0:
            print('==' * 27)
            print("Epoch:{}, average loss {:.4f}, min loss {:.4f}".format(epoch, average_epoch_loss, args.min_loss))



def eval_high(vali_window_num, pos, neg, embeddings=None):
    auc_list, auc_pr_list, ap_list = [], [], []
    embeddings = embeddings.detach()
    for t in range(vali_window_num):
        feature_pooling, order_pooling = pooling_method1(args.ratio, embeddings, pos[t], neg[t], len(pos[t]))
        order_embedding, order_labels = perturb_feature(args.seed, feature_pooling, order_pooling)
        MLP_output = model.forward4(order_embedding.to(torch.float32))
        AUC_value, AUC_PR_value, AP_value = metrics(MLP_output.detach().cpu(), order_labels)
        auc_list.append(AUC_value)
        auc_pr_list.append(AUC_PR_value)
        ap_list.append(AP_value)
    return np.mean(auc_list), np.mean(ap_list)


def metrics(MLP_output, order_labels):
    AUC_value = roc_auc_score(order_labels, MLP_output)
    precision, recall, thresholds = precision_recall_curve(order_labels, MLP_output)
    AUC_PR_value = auc(recall, precision)
    AP_value = average_precision_score(order_labels, MLP_output)
    return AUC_value, AUC_PR_value, AP_value


if __name__ == '__main__':
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print(args)
    # Load data
    clus, degree, inci_matrix, adj_train, high_adj, pos_train, neg_train, pos_test, neg_test, \
      node_number, test_window_num, lap_p, lap_n = load_data(args.ratio, args.seed, args.tau, args.time_num, args.dataset)
    # Initialize the model
    model = DHP(nodes=node_number,
                nfeat=args.initial_dim,
                nhid1=args.hidden_dim1,
                noutput=args.output_dim,
                dropout=args.dropout,
                n_feature=1 * args.output_dim,
                n_hidden=args.mlp_hidden_dim,
                n_output=args.mlp_output_dim,
                hidden_size=args.att_hidden_size,
                loss_decay=args.loss_decay,
                seed=args.seed,
                ratio=args.ratio,
                loss_temp=args.loss_temp)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Train and test
    train()
    print('======================')
    print("Testing Finished!")




