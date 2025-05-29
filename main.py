import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import time
from datasets import get_data_loader
from models.resnet_cifar10 import ResNet18_CIFAR10
import os

def train(model, device, train_loader, optimizer, criterion, scaler, use_amp_effective):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train_samples = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp_effective):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Calculate training accuracy for the batch
        pred_train = output.argmax(dim=1, keepdim=True)
        correct_train += pred_train.eq(target.view_as(pred_train)).sum().item()
        total_train_samples += data.size(0)
            
    end_time = time.time()
    epoch_duration = end_time - start_time
    samples_per_second = total_train_samples / epoch_duration if epoch_duration > 0 else 0
    
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct_train / total_train_samples if total_train_samples > 0 else 0
    
    return avg_loss, train_accuracy, samples_per_second

def test(model, device, test_loader, criterion, use_amp_effective):
    model.eval()
    test_loss_sum = 0
    correct = 0
    total_test_samples = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            total_test_samples += data.size(0)
            with autocast(enabled=use_amp_effective):
                output = model(data)
                loss = criterion(output, target) 
            test_loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    end_time = time.time()
    test_duration = end_time - start_time
    test_samples_per_second = total_test_samples / test_duration if test_duration > 0 else 0

    avg_test_loss = test_loss_sum / total_test_samples if total_test_samples > 0 else 0
    accuracy = 100. * correct / total_test_samples if total_test_samples > 0 else 0
    
    return avg_test_loss, accuracy, test_samples_per_second

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training Demo for ResNet-18 with Mixed Precision and TensorBoard')
    parser.add_argument('-b','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda id (default: 0, use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset-path', type=str, default='/home/chenzh/cifar-10/',
                        help='Path to CIFAR-10 dataset root directory')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (AMP) training')
    parser.add_argument('--log-dir', type=str, default='logs/cifar10_experiment',
                        help='Directory for TensorBoard logs (default: logs/cifar10_experiment)')

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp_effective = args.amp and (device.type == 'cuda')

    if use_amp_effective:
        print("Automatic Mixed Precision (AMP) training effectively enabled.")
    else:
        if args.amp and device.type == 'cpu':
            print("AMP was requested but is only effective on CUDA. Running on CPU without AMP.")
        else:
            print("AMP disabled or not requested.")

    writer = SummaryWriter(log_dir=args.log_dir)

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loader(dataset_type='cifar10', 
                                                img_size=32, 
                                                train_batch_size=args.batch_size,
                                                test_batch_size=args.test_batch_size) 
    print("Dataset loaded.")

    print("Initializing model...")
    model = ResNet18_CIFAR10(num_classes=10).to(device)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler(enabled=use_amp_effective)

    print("Starting training...")
    max_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_speed = train(model, device, train_loader, optimizer, criterion, scaler, use_amp_effective)
        test_loss, test_acc, test_speed = test(model, device, test_loader, criterion, use_amp_effective)
        scheduler.step()
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True
        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': test_acc
        }
        print(f"Epoch {epoch} Summary: \n"
              f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Speed={train_speed:.2f} samples/sec\n"
              f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%, Speed={test_speed:.2f} samples/sec")
        if save_max:
            torch.save(checkpoint, os.path.join(args.log_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(args.log_dir, 'checkpoint_latest.pth'))

    print("Training finished.")
    writer.close()

if __name__ == '__main__':
    main()
