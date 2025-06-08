import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
from datasets import get_data_loader
from models.resnet_cifar10 import ResNet18_CIFAR10
from models.tiny_vgg import TinyVGG
import os

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train_samples = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model.quantize_forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
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

def q_inference(model, device, test_loader, criterion):
    model.eval()
    test_loss_sum = 0
    correct = 0
    total_test_samples = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            total_test_samples += data.size(0)
            output = model.quantize_inference_integer(data)
            loss = criterion(output, target)
            test_loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    end_time = time.time()
    test_duration = end_time - start_time
    test_samples_per_second = total_test_samples / test_duration if test_duration > 0 else 0

    avg_test_loss = test_loss_sum / total_test_samples if total_test_samples > 0 else 0
    accuracy = 100. * correct / total_test_samples if total_test_samples > 0 else 0
    print(f"Integer model Test:  Loss={avg_test_loss:.4f}, Acc={accuracy:.2f}%, Speed={test_samples_per_second:.2f} samples/sec")


def main():
    parser = argparse.ArgumentParser(description='LSQ W4A4 Quantization Inference')
    parser.add_argument('-b','--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('-tb','--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-cuda','--cuda', type=int, default=0, help='cuda id (default: 0, use -1 for CPU)')
    parser.add_argument('-w','--w-num-bits', type=int, default=4, help='Number of bits for weight quantization (default: 4 for W4A4)')
    parser.add_argument('-a','--a-num-bits', type=int, default=4, help='Number of bits for activation quantization (default: 4 for W4A4)')
    parser.add_argument('-d','--dataset', type=str, default='cifar10', help='Dataset to use (default: cifar10)')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model to use (default: resnet18)')
    parser.add_argument('--image-size', type=int, default=32,
                        help='Image size (default: 32)')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes (default: 10)')

    args = parser.parse_args()
    print(args)

    device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    train_loader, test_loader = get_data_loader(dataset_type=args.dataset, 
                                                img_size=args.image_size, 
                                                train_batch_size=args.batch_size,
                                                test_batch_size=args.test_batch_size) 
    print("Dataset loaded.")

    if args.model == 'resnet_cifar10':
        model = ResNet18_CIFAR10(num_classes=args.num_classes).to(device)
    elif args.model == 'tiny_vgg':
        model = TinyVGG(image_size=args.image_size, num_classes=args.num_classes)
    else:
        raise ValueError(f"Network {args.model} not supported.")
    
    criterion = nn.CrossEntropyLoss()

    print("Fusing Batch Norm layers...")
    model.eval()
    model.fuse_bn()
    print("BN layers fused.")

    print(f"Quantizing model to W{args.w_num_bits}A{args.a_num_bits}...")
    model.quantize(w_num_bits=args.w_num_bits, a_num_bits=args.a_num_bits)
    print("Model quantized.")
    state_dict = torch.load(f'./qat_logs/{args.dataset}/{args.model}_w{args.w_num_bits}a{args.a_num_bits}/checkpoint_max.pth', map_location='cpu')['net']
    model.load_state_dict(state_dict)
    model.to(device)
    model.freeze()
    print("Model frozen.")
    with torch.no_grad():
        q_inference(model, device, test_loader, criterion)

    print("Quantization-aware inference finished.")


if __name__ == '__main__':
    main()
