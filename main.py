from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import AmazonDataset as dataset

# This is the network structure
class Net(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super(Net, self).__init__()
        
        self.feature_size = vocab_size
        self.embed_dim = embed_dim

        self.embed1 = nn.Linear(in_features = vocab_size, out_features = embed_dim)
        # self.embed2 = nn.Linear(in_features = embed_dim, out_features = embed_dim)
        self.bn_x = nn.BatchNorm1d(num_features=embed_dim)
        self.bn_y = nn.BatchNorm1d(num_features=embed_dim)
        self.cos = nn.CosineSimilarity()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        x = self.bn_x(self.embed1(x))
        y = self.bn_y(self.embed1(y))
        #x = self.embed(x)
        #y = self.embed(y)

        x = self.tanh(x)
        y = self.tanh(y)

        r = self.cos(x,y)

        return r
        # Let's have a softmax with 
    def forward_query_embed(self, x):

        x = self.bn_x(self.embed1(x))
        x = self.tanh(x)

        return x
    def forward_asin_embed(self, y):
        y = self.bn_y(self.embed1(y))
        y = self.tanh(y)

        return y

def extend_hinge_loss (output, target):
    return - torch.sum(torch.mul(output, target))


def precision_at_k(ground_truth_batch, predictions_batch, reduction='mean'):
    """

    :param ground_truth_batch:
    :param predictions_batch:
    :param reduction:
    :return:
    """
    k_max = 5
    _,  indices = torch.sort(predictions_batch, descending=True)
    top_k = indices[:,:k_max]
    top_k_in_ground_truth = torch.gather(ground_truth_batch, 1, top_k)
    n = ground_truth_batch.shape[0]
    if reduction == 'mean':
        precision_5 = top_k_in_ground_truth[:, :5].gt(0).sum().item() / (5 * n)
        precision_3 = top_k_in_ground_truth[:, :3].gt(0).sum().item() / (3 * n)
        precision_1 = top_k_in_ground_truth[:, :1].gt(0).sum().item() / (1 * n)
    elif reduction == 'sum':
        precision_5 = top_k_in_ground_truth[:, :5].gt(0).sum().item() / (5)
        precision_3 = top_k_in_ground_truth[:, :3].gt(0).sum().item() / (3)
        precision_1 = top_k_in_ground_truth[:, :1].gt(0).sum().item() / (1)
    else:
        precision_5 = top_k_in_ground_truth[:, :5].gt(0).sum(1) / 5
        precision_3 = top_k_in_ground_truth[:, :3].gt(0).sum(1) / 3
        precision_1 = top_k_in_ground_truth[:, :1].gt(0).sum(1) / 1
    return precision_1, precision_3, precision_5


def evaluate(args, model, device, test_loader, iterate_all=False):
    """

    :param args:
    :param model:
    :param device:
    :param test_loader:
    :param iterate_all:
    :return:
    """
    model.eval()
    test_loss = 0.0
    test_precision_1 = 0.0
    test_precision_3 = 0.0
    test_precision_5 = 0.0
    batch_num_small = 50
    batch_num = 0
    for batch_idx, batch in enumerate(test_loader):
        # print(batch)
        data = torch.sparse.FloatTensor(torch.LongTensor(batch['feature_index']),
                                        torch.Tensor(batch['feature_value']),
                                        torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
        target = torch.sparse.FloatTensor(torch.LongTensor(batch['label_index']),
                                          torch.Tensor(batch['label_value']),
                                          torch.Size([args.batch_size, model.label_size])).to(device).to_dense()
        output = model(data)
        # test_loss += nn.functional.binary_cross_entropy(output, target).item()
        test_loss += extend_hinge_loss(output, target).item()
        test_p1, test_p3, test_p5 = precision_at_k(target, output, 'mean')
        test_precision_1 += test_p1
        test_precision_3 += test_p3
        test_precision_5 += test_p5
        if (not iterate_all) and (batch_idx == batch_num_small - 1):
            batch_num = batch_num_small
            break
        else:
            batch_num = batch_idx + 1

    test_loss = test_loss / batch_num
    test_precision_1 = test_precision_1 / batch_num
    test_precision_3 = test_precision_3 / batch_num
    test_precision_5 = test_precision_5 / batch_num
    if iterate_all:
        print('**** Whole Test\tLoss: {:.6f}\tPrecision@1:{:.4f}\tPrecision@3:{:.4f}\tPrecision@5:{:.4f}'
              .format(test_loss, test_precision_1, test_precision_3, test_precision_5))
    else:
        print('---- Sampled Test\tLoss: {:.6f}\tPrecision@1:{:.4f}\tPrecision@3:{:.4f}\tPrecision@5:{:.4f}'
              .format(test_loss, test_precision_1, test_precision_3, test_precision_5))
    return test_loss, test_precision_1, test_precision_3, test_precision_5


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
         
        data_q = torch.sparse.FloatTensor(torch.LongTensor(batch['query_feature_index_batch']),
                                                         torch.Tensor(batch['query_feature_value_batch']),
                                                         torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
        data_t = torch.sparse.FloatTensor(torch.LongTensor(batch['title_feature_index_batch']),
                                                         torch.Tensor(batch['title_feature_value_batch']),
                                                         torch.Size([args.batch_size, model.feature_size])).to(device).to_dense()
        '''
        print(batch['query_feature_index_batch'])
        print(batch['title_feature_index_batch'])
        data_q = torch.LongTensor(batch['query_feature_index_batch']).to(device)
        data_t = torch.LongTensor(batch['title_feature_index_batch']).to(device)
        ''' 
        target = torch.FloatTensor(batch['label']).to(device)
        #target[target == -1] = 0
        #target = target.to(device)

        optimizer.zero_grad()
        output = model(data_q,data_t)

        #print (output)
        #print (target)
        loss = extend_hinge_loss(output,target)
        #loss = F.binary_cross_entropy(output, target)
        #loss = F.hinge_embedding_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train-file', type=str, metavar='N',
                        help='The Path of the training data')
    parser.add_argument('--test-file', type=str, metavar='N',
                        help='The Path of the testing data')
    parser.add_argument('--vocab-size', type=int, default=23486, metavar='N',
                        help='The Vocab size of the data')
    parser.add_argument('--embed-dim', type=int, default=256, metavar='N',
                        help='The dimension of the Embedding vector')
   
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset(args.train_file),
        batch_size=args.batch_size, shuffle=True, collate_fn=dataset.AmazonDataset_collate, drop_last = True)
    test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset(args.test_file),
        batch_size=args.test_batch_size, shuffle=True, collate_fn=dataset.AmazonDataset_collate, drop_last = True)


    model = Net(embed_dim = args.embed_dim, vocab_size = args.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"embed.pt")
        
if __name__ == '__main__':
    main()
