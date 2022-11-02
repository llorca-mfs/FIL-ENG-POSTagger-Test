import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datautils

from tqdm import tqdm

from utils.utils import predict, normalize, produce_vocab, proc_set, init_weights, accuracy
from utils.model import LSTMTagger

import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='Train a part of speech tagger.')
    parser.add_argument('--do_predict', action='store_true', help='Use a trained model to predict parts of speech.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='Location to save model.')
    parser.add_argument('--overwrite_save_directory', action='store_true', help='Overwrite the save directory if it exists.')

    parser.add_argument('--train_data', type=str, help='Training text dataset.')
    parser.add_argument('--evaluation_data', type=str, help='Evaluation text dataset.')
    parser.add_argument('--train_tags', type=str, help='Training tags dataset.')
    parser.add_argument('--evaluation_tags', type=str, help='Evaluation tags dataset.')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use a GPU.')
    
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of recurrent layers.')
    parser.add_argument('--bidirectional', action='store_true', help='Use a bidirectional RNN.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--recur_dropout', type=float, default=0.1, help='Recurrent dropout probability.')
    parser.add_argument('--min_freq', type=int, default=1, help='Minimum frequency of words to be added to vocabulary.')
    parser.add_argument('--msl', type=int, default=128, help='Maximum sequence length of text.')
    parser.add_argument('--bs', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--sentence', type=str, default='Hello', help='Sentence to predict')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed);
    device = torch.device('cpu')

    if args.do_train:
        # Load Dataset
        print("Loading dataset")
        with open(args.train_data, 'r') as f:
            train_words = [line.strip() for line in f]
        with open(args.evaluation_data, 'r') as f:
            test_words = [line.strip() for line in f]
        with open(args.train_tags, 'r') as f:
            train_tags = [line.strip() for line in f]
        with open(args.evaluation_tags, 'r') as f:
            test_tags = [line.strip() for line in f]

        # Normalize text
        print("Normalizing text and producing vocabularies.")
        train_words = [normalize(line) for line in train_words]
        test_words = [normalize(line) for line in test_words]

        # Produce vocabularies
        word_vocab, idx2word, word2idx = produce_vocab(train_words, min_freq=args.min_freq)
        tags_vocab, idx2tag, tag2idx  = produce_vocab(train_tags, min_freq=args.min_freq)
        print("Training word vocabulary has {:,} unique tokens.".format(len(word_vocab)))
        print("Training tags vocabulary has {:,} unique tokens.".format(len(tags_vocab)))

        # Produce sets
        X_train = proc_set(train_words, word2idx, word_vocab, msl=args.msl)
        y_train = proc_set(train_tags , tag2idx,  tags_vocab,  msl=args.msl)
        X_test = proc_set(test_words, word2idx, word_vocab, msl=args.msl)
        y_test = proc_set(test_tags , tag2idx,  tags_vocab,  msl=args.msl)

        # Convert to tensors
        X_train, y_train = torch.LongTensor(X_train), torch.LongTensor(y_train)
        X_test, y_test = torch.LongTensor(X_test), torch.LongTensor(y_test)

        # Produce dataloaders
        train_set = datautils.TensorDataset(X_train, y_train)
        test_set = datautils.TensorDataset(X_test, y_test)
        train_sampler = datautils.RandomSampler(train_set)
        train_loader = datautils.DataLoader(train_set, sampler=train_sampler, batch_size=args.bs)
        test_loader = datautils.DataLoader(test_set, shuffle=False, batch_size=args.bs)

        print("Training batches: {}\nEvaluation batches: {}".format(len(train_loader), len(test_loader)))

        # Training setup
        model = LSTMTagger(word_vocab_sz=len(word_vocab), 
                           tag_vocab_sz=len(tags_vocab), 
                           embedding_dim=args.embedding_dim, 
                           hidden_dim=args.hidden_dim, 
                           dropout=args.dropout,
                           num_layers=args.num_layers,
                           recur_dropout=args.recur_dropout,
                           bidirectional=args.bidirectional).to(device)
        model.apply(init_weights)
        criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['<pad>'])
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        print("Model has {:,} trainable parameters.".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))   

        # Training
        for e in range(1, args.epochs + 1):
            model.train()
            train_loss, train_acc = 0, 0
            for x, y in tqdm(train_loader):
                x, y = x.transpose(1, 0).to(device), y.transpose(1, 0).to(device)
                out = model(x)
                loss = criterion(out.flatten(0, 1), y.flatten(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += accuracy(out, y, tag2idx)
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            model.eval()
            test_loss, test_acc = 0, 0
            for x, y in tqdm(test_loader):
                with torch.no_grad():
                    x, y = x.transpose(1, 0).to(device), y.transpose(1, 0).to(device)
                    out = model(x)
                    loss = criterion(out.flatten(0, 1), y.flatten(0))
                test_loss += loss.item()
                test_acc += accuracy(out, y, tag2idx)
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

            print("Epoch {:4} | Train Loss {:.4f} | Train Acc {:.2f}% | Test Loss {:.4f} | Test Acc {:.2f}%".format(e, train_loss, train_acc, test_loss, test_acc))  
        
        # Save model
        if args.overwrite_save_directory:
            if os.path.exists(args.checkpoint): os.system('rm -r '+ args.checkpoint + '/')

        print('Saving model and vocabularies.')
        os.mkdir(args.checkpoint)
        with open(args.checkpoint + '/model.bin', 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(args.checkpoint + '/settings.bin', 'wb') as f:
            torch.save([word_vocab, word2idx, idx2word, tags_vocab, tag2idx, idx2tag, args.msl, 
                        args.embedding_dim, args.hidden_dim, args.dropout, args.bidirectional, 
                        args.num_layers, args.recur_dropout], f)

    if args.do_predict:
        # Load the vocabularies
        with open(args.checkpoint + '/settings.bin', 'rb') as f:
            word_vocab, word2idx, idx2word, tags_vocab, tag2idx, idx2tag, msl, embedding_dim, hidden_dim, dropout, bidirectional, num_layers, recur_dropout = torch.load(f)

        # Produce a blank model
        model = LSTMTagger(word_vocab_sz=len(word_vocab), 
                           tag_vocab_sz=len(tags_vocab), 
                           embedding_dim=embedding_dim, 
                           hidden_dim=hidden_dim, 
                           dropout=dropout,
                           num_layers=num_layers,
                           recur_dropout=recur_dropout,
                           bidirectional=bidirectional)

        # Load checkpoints and put the model in eval mode
        with open(args.checkpoint + '/model.bin', 'rb') as f:
            model.load_state_dict(torch.load(f))
        model = model.cpu()
        model.eval();

        preds = predict(args.sentence, word2idx, idx2tag, word_vocab, msl, model)
        print(preds)

if __name__ == '__main__':
    main()
