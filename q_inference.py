import torch
import argparse
import os
import time

from datasets import get_data_loader
from models.resnet_cifar10 import ResNet18_CIFAR10

def inference(args):
    device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    _, test_loader = get_data_loader(dataset_type=args.dataset,
                                     img_size=32,  # Assuming CIFAR-10 default
                                     train_batch_size=1, # Not used
                                     test_batch_size=args.test_batch_size)
    print("Dataset loaded.")

    # Load checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist.")
        return
    
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu') # Load to CPU first

    w_num_bits = checkpoint.get('w_num_bits')
    a_num_bits = checkpoint.get('a_num_bits')

    if w_num_bits is None or a_num_bits is None:
        print("Error: w_num_bits or a_num_bits not found in checkpoint. Cannot proceed with quantization setup.")
        # Fallback or allow user to specify if not in checkpoint
        # For now, let's assume they must be in the checkpoint as per q_main.py
        w_num_bits = args.w_num_bits_fallback # Add this as an arg if needed
        a_num_bits = args.a_num_bits_fallback # Add this as an arg if needed
        print(f"Warning: Using fallback w_bits={w_num_bits}, a_bits={a_num_bits}. Inference might be incorrect if these don't match training.")
        # return

    # Initialize model
    print(f"Initializing model ({args.net}) for W{w_num_bits}A{a_num_bits} integer-only inference...")
    if args.net == 'resnet_cifar10':
        # Create a float model instance first
        model = ResNet18_CIFAR10(num_classes=10) # Assuming 10 classes for CIFAR-10
    else:
        raise ValueError(f"Network {args.net} not supported.")

    # Fuse BN layers (must be done on the float model before quantization structure is applied)
    model.eval() # Set to eval mode for BN fusion
    print("Fusing Batch Norm layers...")
    model.fuse_bn()
    print("BN layers fused.")

    # Apply quantization structure
    print(f"Applying quantization structure W{w_num_bits}A{a_num_bits} to the model...")
    model.quantize(w_num_bits=w_num_bits, a_num_bits=a_num_bits)
    print("Quantization structure applied.")

    # Load the quantized state_dict
    model.load_state_dict(checkpoint['net'])
    print("Loaded quantized model weights from checkpoint.")
    
    model.to(device)
    model.eval() # Ensure model is in evaluation mode

    # Freeze the model for integer-only inference (calculates M, n, etc.)
    print("Freezing model for pure integer inference...")
    model.freeze()
    print("Model frozen.")

    correct = 0
    total = 0
    
    print("Starting inference...")
    inference_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # Perform pure integer inference
            output = model.quantize_inference(data)
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed [{batch_idx + 1}/{len(test_loader)}] batches...")

    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time
    
    accuracy = 100. * correct / total
    print(f"Inference finished in {inference_duration:.2f} seconds.")
    print(f"Test Accuracy on {args.dataset} (W{w_num_bits}A{a_num_bits} pure integer inference): {accuracy:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Pure Integer Inference for ResNet')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA ID (default: 0, use -1 for CPU)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='dataset to use (default: cifar10)')
    parser.add_argument('--net', type=str, default='resnet_cifar10',
                        help='network to use (default: resnet_cifar10)')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to the trained quantized model checkpoint (.pth file from q_main.py)')
    # Fallback quantization bits if not found in checkpoint (optional, but good for robustness)
    parser.add_argument('--w-num-bits-fallback', type=int, default=4, help='Fallback weight bits if not in checkpoint')
    parser.add_argument('--a-num-bits-fallback', type=int, default=4, help='Fallback activation bits if not in checkpoint')


    args = parser.parse_args()
    inference(args)

if __name__ == '__main__':
    main()
'''
python q_inference.py --checkpoint-path qat_logs/resnet_cifar10_w4a4/checkpoint_max_w4a4.pth
'''