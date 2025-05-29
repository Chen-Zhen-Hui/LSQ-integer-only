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
            output = model.quantize_forward(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
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
                output = model.quantize_forward(data)
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

def test_float(model, device, test_loader, criterion, use_amp_effective):
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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 W4A4 Quantization Training for ResNet-18') # MODIFIED: Description
    parser.add_argument('-b','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-e','--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda id (default: 0, use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed')
    parser.add_argument('--amp', '-amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (AMP) training (can be unstable with QAT)')
    parser.add_argument('--log-dir', type=str, default='qat_logs', help='Directory for TensorBoard logs')
    parser.add_argument('--w-num-bits', type=int, default=4, help='Number of bits for weight quantization (default: 4 for W4A4)')
    parser.add_argument('--a-num-bits', type=int, default=4, help='Number of bits for activation quantization (default: 4 for W4A4)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use (default: cifar10)')
    parser.add_argument('--net', type=str, default='resnet_cifar10', help='Network to use (default: resnet_cifar10)')

    args = parser.parse_args()
    out_dir = os.path.join(args.log_dir, f'{args.net}_w{args.w_num_bits}a{args.a_num_bits}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp_effective = args.amp and (device.type == 'cuda')

    if use_amp_effective:
        print("Automatic Mixed Precision (AMP) training effectively enabled. Note: AMP with QAT can sometimes be unstable.")
    else:
        if args.amp and device.type == 'cpu':
            print("AMP was requested but is only effective on CUDA. Running on CPU without AMP.")
        else:
            print("AMP disabled or not requested.")

    writer = SummaryWriter(log_dir=out_dir)

    print("Loading dataset...")
    train_loader, test_loader = get_data_loader(dataset_type=args.dataset, 
                                                img_size=32, 
                                                train_batch_size=args.batch_size,
                                                test_batch_size=args.test_batch_size) 
    print("Dataset loaded.")

    print("Initializing model for W{}A{} quantization...".format(args.w_num_bits, args.a_num_bits))
    if args.net == 'resnet_cifar10':
        model = ResNet18_CIFAR10(num_classes=10).to(device)
        model.load_state_dict(torch.load('./logs/cifar10_experiment/checkpoint_max.pth')['net'])
    else:
        raise ValueError(f"Network {args.net} not supported.")
    
    print("Fusing Batch Norm layers...")
    model.eval()
    model.fuse_bn()
    print("BN layers fused.")
    criterion = nn.CrossEntropyLoss()
    # test_loss, test_acc, test_speed = test_float(model, device, test_loader, criterion, use_amp_effective)
    # print(f"Float model Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%, Speed={test_speed:.2f} samples/sec")

    print(f"Quantizing model to W{args.w_num_bits}A{args.a_num_bits}...")
    model.quantize(w_num_bits=args.w_num_bits, a_num_bits=args.a_num_bits)
    print("Model quantized.")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler(enabled=use_amp_effective)

    print("Starting W{}A{} quantization-aware training...".format(args.w_num_bits, args.a_num_bits))
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
            'max_test_acc': test_acc,
            'w_num_bits': args.w_num_bits,
            'a_num_bits': args.a_num_bits
        }
        
        print(f"Epoch {epoch} Summary: (W{args.w_num_bits}A{args.a_num_bits} QAT)\n"
              f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Speed={train_speed:.2f} samples/sec\n"
              f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%, Speed={test_speed:.2f} samples/sec")

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_max_w{args.w_num_bits}a{args.a_num_bits}.pth'))
        torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_latest_w{args.w_num_bits}a{args.a_num_bits}.pth'))

    print("Quantization-aware training finished.")


    writer.close()

if __name__ == '__main__':
    main()
