import os
import ray

# 获取可用的 GPU 数量


ray.init(num_gpus=1)

available_gpus = ray.available_resources().get('GPU', 0)
print(f"Available GPUs: {available_gpus}")

@ray.remote(num_gpus=0.5)  # 请求半个 GPU
class GPUActor:
    def ping(self):
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
        print(f"Actor - CUDA_VISIBLE_DEVICES: {cuda_devices}")
        
        gpu_ids = ray.get_gpu_ids()
        print(f"Actor - Assigned GPU IDs: {gpu_ids}")

@ray.remote(num_gpus=0.5)  
def gpu_task():
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"Task - CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    gpu_ids = ray.get_gpu_ids()
    print(f"Task - Assigned GPU IDs: {gpu_ids}")


gpu_actor = GPUActor.remote()
ray.get(gpu_actor.ping.remote())


ray.get(gpu_task.remote())


ray.shutdown()