import os
from tqdm import trange

import argparse
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import FlowFormer
from util import MetricsCalculator, GlowLoss, preprocess_images

best_loss = 1e15

parser = argparse.ArgumentParser(description="Glow trainer")
# Train settings
# parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--n_samples", default=20, type=int, help="number of samples")
parser.add_argument("--resume", type=str, default=None, help="resume file name")
parser.add_argument("--loss_scale", type=float, default=1.)

# Model Architechture
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--n_attnflow", default=32, 
                    type=int, 
                    help="number of attention+flow blocks in each block")
parser.add_argument("--n_blocks", default=4, 
                    type=int, help="number of blocks")
# Misc
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")


def train(epoch, model: FlowFormer, optimizer, dataloader: DataLoader):
    model.train()
    
    total_loss = 0
    for images, _ in dataloader:
        optimizer.zero_grad()
        images = preprocess_images(images, device, args.n_bits)

        log_p, log_det = model(images)

        loss, log_p, log_det = criterion(log_p, log_det)
        # loss *= args.loss_scale
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss}, epoch)


def test(epoch, model: FlowFormer, dataloader: DataLoader):
    global best_loss
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for images, _ in dataloader:
            images = preprocess_images(images, device, args.n_bits)

            log_p, log_det = model(images)
            loss, log_p, log_det = criterion(log_p, log_det)
            # loss *= args.loss_scale
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    state = {
        "model": model.state_dict(),
        "test_loss": avg_loss,
        "epoch": epoch
    }
    os.makedirs("checkpoints", exist_ok=True)
    last_path = os.path.join("checkpoints", f"vit_{epoch - 5}.pt")
    if os.path.exists(last_path):
        os.remove(last_path)
    torch.save(state, os.path.join("checkpoints", f"vit_{epoch}.pt"))

    sample_images = model.sample(args.n_samples, args.temp)

    wandb.log({"test_loss": avg_loss}, epoch)
    wandb.log({"image_metrics": metrics_calculator.compute(sample_images)}, epoch)

    images_concat = utils.make_grid(sample_images, nrow=args.n_samples // 2, padding=2, pad_value=255, normalize=True, range=(-0.5, 0.5))
    wandb.log({"samples": wandb.Image(transforms.ToPILImage()(images_concat))}, epoch)


def main():
    transform_train = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    trainset = datasets.CIFAR10(root='data', download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    testset = datasets.CIFAR10(root='data', train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    print("Building the model...")
    model = FlowFormer(3, args.img_size, args.n_blocks, num_attnflow=args.n_attnflow)
    global criterion
    criterion = GlowLoss(3, args.img_size, 2**args.n_bits)

    model = model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print(f'Resuming from checkpoint at checkpoints/{args.resume}...')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'checkpoints/{args.resume}')
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Initializing KID and FID..')
    global metrics_calculator
    metrics_calculator = MetricsCalculator(device)
    metrics_calculator.initialize(testloader)

    print("Training...")
    for epoch in trange(start_epoch, args.epochs):
        train(epoch, model, optimizer, trainloader)
        test(epoch, model, testloader)


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="FlowFormer", 
            config={
                "architechture": "FlowFormer",
                "dataset": "CIFAR-10",
                "epochs": args.epochs,
                "learning_rate": args.lr,
                },
            dir="logs")

    main()