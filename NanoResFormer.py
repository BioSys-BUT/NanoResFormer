import argparse
from NanoResFormer_predict import NanoResFormer_predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NanoResFormer inference on signals listed in a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter  # For better help formatting
    )
    
    # --- Required Arguments ---

    parser.add_argument(
        'csv_path', 
        type=str, 
        help='Path to input CSV file containing raw signals.'
    )
    parser.add_argument(
        'out_dir', 
        type=str, 
        help='Directory where results (CSV and optional images) will be saved.'
    )
    
    # --- Optional Arguments ---
    parser.add_argument(
        '--csv_name', 
        type=str, 
        default=None, 
        help='Base name for output files (default: uses input filename).'
    )
    parser.add_argument(
        '--OV', 
        type=int, 
        default=80, 
        help='Window overlap percentage (10-99). Default: 80.'
    )
    parser.add_argument(
        '--Model', 
        type=str, 
        default='middle', 
        choices=['low', 'middle', 'high'],  # Recommended to limit inputs
        help="Model variant to use (e.g., 'middle', 'low', 'high'). Default: 'middle'."
    )
    # For boolean (True/False) use store_true/store_false
    parser.add_argument(
        '--export_images', 
        action='store_true', 
        help='If present, export annotated images for each processed signal. Default: False.'
    )
    parser.add_argument(
        '--device_pref', 
        type=str, 
        default='auto', 
        choices=['auto', 'cpu', 'cuda'],
        help="Device selection preference: 'cpu', 'cuda', or 'auto'. Default: 'auto'."
    )
    
    args = parser.parse_args()
    
    try:
        NanoResFormer_predict(
            csv_path=args.csv_path,
            out_dir=args.out_dir,
            csv_name=args.csv_name,
            OV=args.OV,
            Model=args.Model,
            export_images=args.export_images,
            device_pref=args.device_pref
        )
    except Exception as e:
        print(f"\nERROR: NanoResFormer execution failed. {e}")
