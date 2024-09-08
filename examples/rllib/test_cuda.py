import torch
import sys
import time

def test_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        
        # Get the number of CUDA devices
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # Print information for each CUDA device
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            print(f"\nCUDA Device {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Create a small tensor and move it to GPU
        x = torch.rand(5, 3)
        x = x.to('cuda')
        print(f"\nTensor on GPU: {x}")
        
    else:
        print("CUDA is not available. Running on CPU.")

    # Print PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Print Python version
    print(f"Python version: {sys.version}")

if __name__ == "__main__":
    test_cuda()
    print("Done")
