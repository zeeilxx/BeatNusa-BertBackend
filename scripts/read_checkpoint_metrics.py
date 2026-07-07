import sys
from pathlib import Path
import torch

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_checkpoint_metrics.py <path_to_checkpoint.pt>")
        sys.exit(1)
        
    ckpt_path = Path(sys.argv[1])
    if not ckpt_path.exists():
        print(f"Error: File {ckpt_path} does not exist.")
        sys.exit(1)
        
    try:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"=== Metrics for {ckpt_path.name} ===")
        print(f"Epoch: {state.get('epoch', 'N/A')}")
        print(f"Validation Loss: {state.get('val_loss', 'N/A')}")
        
        metrics = state.get('val_metrics', {})
        if metrics:
            print("Validation Metrics:")
            for k, v in metrics.items():
                print(f"  - {k}: {v:.6f}")
        else:
            print("No validation metrics found in this checkpoint.")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == '__main__':
    main()
