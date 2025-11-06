"""
TensorRT Optimization for NVIDIA Jetson Xavier NX
Convert YOLOv8 to TensorRT engine for real-time inference
"""

import torch
import tensorrt as trt
import onnx
import numpy as np
from pathlib import Path
import argparse
import time
import cv2


class TensorRTOptimizer:
    """
    Optimize YOLOv8 model for Jetson Xavier NX using TensorRT
    """
    
    def __init__(self, onnx_path, engine_path, fp16=True, batch_size=1, img_size=640):
        """
        Initialize TensorRT optimizer
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            fp16: Use FP16 precision
            batch_size: Batch size for inference
            img_size: Input image size
        """
        self.onnx_path = Path(onnx_path)
        self.engine_path = Path(engine_path)
        self.fp16 = fp16
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.logger = trt.Logger(trt.Logger.INFO)
    
    def build_engine(self):
        """
        Build TensorRT engine from ONNX model
        """
        print(f"Building TensorRT engine from {self.onnx_path}")
        print(f"Precision: {'FP16' if self.fp16 else 'FP32'}")
        print(f"Batch size: {self.batch_size}")
        print(f"Input size: {self.img_size}x{self.img_size}")
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        print("Parsing ONNX model...")
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Create builder config
        config = builder.create_builder_config()
        
        # Set memory pool limits (8GB for Xavier NX)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
        
        # Enable FP16 if requested
        if self.fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 mode enabled")
            else:
                print("WARNING: FP16 not supported on this platform")
        
        # Enable CUDA graphs for faster inference
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        # Build engine
        print("Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        print(f"Saving engine to {self.engine_path}")
        with open(self.engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print("Engine built successfully!")
        
        return serialized_engine
    
    def load_engine(self):
        """Load TensorRT engine from file"""
        print(f"Loading TensorRT engine from {self.engine_path}")
        
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        print("Engine loaded successfully!")
        return engine
    
    def benchmark(self, num_iterations=100):
        """
        Benchmark inference performance
        
        Args:
            num_iterations: Number of iterations for benchmarking
        """
        print("\n" + "="*60)
        print("Benchmarking TensorRT Engine")
        print("="*60)
        
        # Load engine
        engine = self.load_engine()
        context = engine.create_execution_context()
        
        # Allocate buffers
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Get input/output shapes
        input_shape = (self.batch_size, 3, self.img_size, self.img_size)
        
        # Allocate host and device buffers
        h_input = np.random.randn(*input_shape).astype(np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        
        # Output buffers (YOLOv8 has multiple outputs)
        outputs = []
        d_outputs = []
        for i in range(1, engine.num_io_tensors):
            shape = engine.get_tensor_shape(engine.get_tensor_name(i))
            output = np.empty(tuple(shape), dtype=np.float32)
            d_output = cuda.mem_alloc(output.nbytes)
            outputs.append(output)
            d_outputs.append(d_output)
        
        # Warm-up
        print("Warming up...")
        for _ in range(10):
            cuda.memcpy_htod(d_input, h_input)
            context.execute_v2([int(d_input)] + [int(d) for d in d_outputs])
            cuda.Context.synchronize()
        
        # Benchmark
        print(f"Running {num_iterations} iterations...")
        torch.cuda.empty_cache()  # Clear CUDA cache
        
        times = []
        for i in range(num_iterations):
            # Copy input to device
            cuda.memcpy_htod(d_input, h_input)
            
            # Run inference
            start = time.perf_counter()
            context.execute_v2([int(d_input)] + [int(d) for d in d_outputs])
            cuda.Context.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
        
        # Calculate statistics
        times = np.array(times) * 1000  # Convert to ms
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000.0 / mean_time
        
        print("\n" + "="*60)
        print("Benchmark Results:")
        print("="*60)
        print(f"Mean inference time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Min inference time:  {min_time:.2f} ms")
        print(f"Max inference time:  {max_time:.2f} ms")
        print(f"FPS:                 {fps:.2f}")
        print(f"Real-time capable:   {'YES ✓' if fps >= 15 else 'NO ✗'}")
        print("="*60 + "\n")
        
        return mean_time, fps


def convert_pytorch_to_onnx(model_path, onnx_path, img_size=640):
    """
    Convert YOLOv8 PyTorch model to ONNX
    
    Args:
        model_path: Path to YOLOv8 .pt model
        onnx_path: Path to save ONNX model
        img_size: Input image size
    """
    from ultralytics import YOLO
    
    print(f"Converting {model_path} to ONNX...")
    
    model = YOLO(model_path)
    onnx_file = model.export(
        format='onnx',
        imgsz=img_size,
        simplify=True,
        dynamic=False,  # Fixed batch size for TensorRT
        opset=12
    )
    
    print(f"ONNX model saved to: {onnx_file}")
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully!")
    
    return onnx_file


def main():
    """Main optimization function"""
    parser = argparse.ArgumentParser(
        description='Optimize YOLOv8 for Jetson Xavier NX with TensorRT'
    )
    
    # Input
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLOv8 .pt model or ONNX model')
    parser.add_argument('--output', type=str, default='yolov8_trt.engine',
                        help='Output TensorRT engine path')
    
    # Optimization settings
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size (640 or 512)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision (default: FP16)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (default: 1 for Jetson)')
    
    # Actions
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark the TensorRT engine')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    
    # Convert to ONNX if input is PyTorch model
    if model_path.suffix == '.pt':
        onnx_path = model_path.with_suffix('.onnx')
        if not onnx_path.exists():
            convert_pytorch_to_onnx(str(model_path), str(onnx_path), args.img_size)
        else:
            print(f"Using existing ONNX model: {onnx_path}")
    else:
        onnx_path = model_path
    
    # Build TensorRT engine
    optimizer = TensorRTOptimizer(
        onnx_path=onnx_path,
        engine_path=args.output,
        fp16=not args.fp32,
        batch_size=args.batch,
        img_size=args.img_size
    )
    
    if not Path(args.output).exists():
        optimizer.build_engine()
    else:
        print(f"Using existing TensorRT engine: {args.output}")
    
    # Benchmark if requested
    if args.benchmark:
        optimizer.benchmark(num_iterations=args.iterations)


if __name__ == "__main__":
    main()
