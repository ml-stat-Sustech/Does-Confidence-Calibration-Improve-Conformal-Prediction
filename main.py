"""
Main entry point for running conformal prediction experiments.

Example usage:
    python main.py --preprocess confts

Prerequisites:
    1. Copy .env.example to .env
    2. Set DATA_DIR in .env to your ImageNet root directory
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file at startup
load_dotenv()

from src.utils import set_seed, build_model, build_dataloader_imagenet, build_preprocessor
from src.conformal import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description='Does-Confidence-Calibration-Improve-Conformal-Prediction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 'densenet121', 'vgg16', 'vit'],
                        help='Model name')
    parser.add_argument('--conformal', type=str, default='aps', help='Conformal prediction method')
    parser.add_argument('--alpha', type=float, default=0.1, help='Error rate (significance level)')
    parser.add_argument('--cal_num', type=int, default=10000, help='Calibration set size')
    parser.add_argument('--conf_num', type=int, default=5000, help='Conformal calibration size')
    parser.add_argument('--temp_num', type=int, default=5000, help='Temperature calibration size')
    parser.add_argument('--preprocess', type=str, default='confts',
                        choices=['confts', 'confps', 'confvs', 'ts', 'ps', 'vs', 'none'],
                        help='Preprocessing method')
    parser.add_argument('--penalty', type=float, default=0.001, help='RAPS penalty parameter')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    return parser.parse_args()


def main():
    args = parse_args()

    PREPROCESS = {
        "confts": "Conformal Temperature Scaling (proposed method)",
        "confps": "Conformal Platt Scaling (proposed method)",
        "confvs": "Conformal Vector Scaling (proposed method)",
        "ts": "Temperature Scaling (baseline method)",
        "ps": "Platt Scaling (baseline method)",
        "vs": "Vector Scaling (baseline method)",
        "none": "Identity (no calibration)",
    }
    
    # Print experiment configuration
    print("\n" + "="*50)
    print("Experiment Configuration:")
    print(f"  Dataset: imagenet")
    print(f"  Model: {args.model}")
    print(f"  Conformal: {args.conformal}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Cal_num: {args.cal_num}")
    print(f"  Conf_num: {args.conf_num}")
    print(f"  Temp_num: {args.temp_num}")
    print(f"  Preprocess: {PREPROCESS[args.preprocess]}")
    if args.conformal.lower() == 'raps':
        print(f"  Penalty: {args.penalty}")
    print("="*50 + "\n")
    
    # Load data directory from environment variable
    data_dir = os.getenv('DATA_DIR')
    if data_dir is None:
        raise ValueError(
            "DATA_DIR not set. Please:\n"
            "  1. Copy .env.example to .env: cp .env.example .env\n"
            "  2. Edit .env and set DATA_DIR to your ImageNet root\n"
            f"  3. Current .env file location: {os.path.abspath('.env')}"
        )
    
    # Set random seed
    set_seed(args.seed)
    
    # Build model
    model = build_model(args.model)
    model = model.cuda()
    
    # Build dataloaders
    calib_calibloader, conf_calibloader, testloader = build_dataloader_imagenet(
        data_dir, conf_num=args.conf_num, temp_num=args.temp_num,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Select preprocessor
    preprocessor = build_preprocessor(args.preprocess, model=model, alpha=args.alpha)
    
    # Build predictor and run experiment
    predictor = Predictor(
        model, preprocessor, args.conformal, args.alpha, 
        random=True, penalty=args.penalty
    )
    ece_before, ece_after = predictor.calibrate(calib_calibloader, conf_calibloader)
    result = predictor.evaluate(testloader)
    
    # Print results
    print("\n" + "="*50)
    print("Results:")
    print(f"  ECE before calibration: {ece_before}")
    print(f"  ECE after calibration:  {ece_after}")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()
