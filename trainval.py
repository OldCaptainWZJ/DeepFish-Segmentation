import argparse
import numpy as np
import torch
from src import datasets, models, training
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot(epoch_list, train_loss_list, val_loss_list, train_score_list, val_score_list, model_dir, save=False):
    # Plot loss curve
    plt.plot(epoch_list, train_loss_list, label="Training")
    plt.plot(epoch_list, val_loss_list, label="Validation")

    plt.title('Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    if not save:
        plt.show()
    else:
        fig_path = os.path.join(model_dir, "loss_plot.png")
        plt.savefig(fig_path)

    plt.clf()

    # Plot score curve
    plt.plot(epoch_list, train_score_list, label="Training")
    plt.plot(epoch_list, val_score_list, label="Validation")

    plt.title('Score Plot')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')

    plt.legend()

    if not save:
        plt.show()
    else:
        fig_path = os.path.join(model_dir, "score_plot.png")
        plt.savefig(fig_path)


def save_checkpoint_state(path, epoch, model, optimizer, scheduler, best_score, best_epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score,
        "best_epoch": best_epoch
    }

    # torch.save(model.state_dict(), os.path.join(path, "model_state_dict.pth"))
    torch.save(checkpoint, os.path.join(path, "checkpoint.tar"))


def load_checkpoint_state(path, device, model, optimizer, scheduler):
    path = os.path.join(path, "checkpoint.tar")

    checkpoint = torch.load(path, map_location=device)

    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_score = checkpoint["best_score"]
    best_epoch = checkpoint["best_epoch"]

    return epoch, model, optimizer, scheduler, best_score, best_epoch


def train(args):
    # set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    if args.resize:
        train_data = datasets.get_dataset('deepfish_seg', 'train', 'resize_normalize', args.data_dir)
        val_data = datasets.get_dataset('deepfish_seg', 'val', 'resize_normalize', args.data_dir)
    else:
        train_data = datasets.get_dataset('deepfish_seg', 'train', 'rgb_normalize', args.data_dir)
        val_data = datasets.get_dataset('deepfish_seg', 'val', 'rgb_normalize', args.data_dir)

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=args.batch_size)
    val_loader = DataLoader(val_data,
                            shuffle=False,
                            batch_size=args.batch_size)

    model = models.get_model(args.network).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

    # Checkpointing
    # =============
    df = None
    s_epoch = 0
    best_score = 0.
    best_epoch = 0

    if not args.retrain:
        # resume experiment
        s_epoch, model, optimizer, scheduler, best_score, best_epoch = load_checkpoint_state(args.model_dir, device, model, optimizer, scheduler)
        df = pd.read_csv(os.path.join(args.model_dir, 'data.csv'))
    elif args.network == "unet":
        # randomly initialize weights
        models.init_weights(model)

    model_params = models.params_init(args)

    train_loss_list = []
    val_loss_list = []
    train_score_list = []
    val_score_list = []

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    # Training loop
    for epoch in range(s_epoch+1, s_epoch+args.epochs+1):
        print('Epoch %d in progress:' % epoch)
        print('Training...')
        train_loss, train_score = training.train_on_loader(model, optimizer, train_loader, args.resize, model_params)
        print('Validating...')
        val_loss, val_score = training.val_on_loader(model, val_loader, args.resize, 'val', model_params)

        train_loss_list.append(train_loss)
        train_score_list.append(train_score)
        val_loss_list.append(val_loss)
        val_score_list.append(val_score)

        scheduler.step()

        # Save checkpoint with best val score
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            save_path = os.path.join(args.model_dir, "best_val_score")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_checkpoint_state(save_path, epoch, model, optimizer, scheduler, best_score, best_epoch)

        # Save checkpoints regularly with set interval
        if (epoch - s_epoch) % args.save_interval == 0:
            save_path = os.path.join(args.model_dir, "epoch_%d" % epoch)
            os.mkdir(save_path)
            save_checkpoint_state(save_path, epoch, model, optimizer, scheduler, best_score, best_epoch)

    # Save checkpoint at the end of training
    save_checkpoint_state(args.model_dir, s_epoch+args.epochs, model, optimizer, scheduler, best_score, best_epoch)

    print("Best val score: %f at Epoch %d" % (best_score, best_epoch))

    # Save training data as csv
    data = {'Train Loss': train_loss_list,
            'Val Loss': val_loss_list,
            'Train Score': train_score_list,
            'Val Score': val_score_list}
    new_df = pd.DataFrame(data)

    if df is None:
        df = new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(os.path.join(args.model_dir, 'data.csv'), index=False)

    # Plot all training data
    epoch_list = df.index.to_list()
    train_loss_list = df['Train Loss'].to_list()
    val_loss_list = df['Val Loss'].to_list()
    train_score_list = df['Train Score'].to_list()
    val_score_list = df['Val Score'].to_list()
    plot(epoch_list, train_loss_list, val_loss_list, train_score_list, val_score_list, args.model_dir, args.save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and validate model')

    parser.add_argument('-d', '--data_dir', required=True, help="Dataset directory")
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Epochs for training and validating")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="Batch size for training and validating")
    parser.add_argument('-re', '--retrain', type=bool, default=False, help="Whether to retrain from scratch")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('-m', '--model_dir', default=None, help="Experiment directory (where you put checkpoints)")
    parser.add_argument('-n', '--network', default="unet", help="Network structure used for experiment")
    parser.add_argument('-rs', '--resize', default=True, help="Whether to resize images before input")
    parser.add_argument('-sv', '--save_plot', default=True, help="Whether to save plot")
    parser.add_argument('-si', '--save_interval', default=100, help="Save checkpoint once every [x] epochs")

    args = parser.parse_args()

    train(args)
