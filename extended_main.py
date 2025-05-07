import os
import argparse
import torch
from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from arguments import args_parser

# Import our new models
from attention_medfuse import AttentionMedFuse, TransformerMedFuse, TemporalAlignmentModule, CrossModalGenerator

# Import trainers
from trainers.fusion_trainer import FusionTrainer
from trainers.mmtm_trainer import MMTMTrainer
from trainers.daft_trainer import DAFTTrainer
from trainers.attention_trainer import AttentionFusionTrainer
from trainers.transformer_trainer import TransformerTrainer

def main():
    parser = args_parser()
    args = parser.parse_args()
    
    # Set optimal uni-modal ratio based on task if not specified
    if args.optimal_uni_ratio is None:
        if args.task == 'in-hospital-mortality':
            args.optimal_uni_ratio = 0.1  # 10% for mortality
        else:
            args.optimal_uni_ratio = 0.2  # 20% for phenotyping
    
    args.data_ratio = args.optimal_uni_ratio
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up data preprocessing
    discretizer = Discretizer(timestep=float(args.timestep),
                             store_masks=True,
                             impute_strategy='previous',
                             start_time='zero')
    
    # Get sample data for discretizer
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    sample_data = np.stack(ret)
    
    # Set up discretizer and normalizer
    discretizer_header = discretizer.transform(sample_data)[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    
    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = f'normalizers/ph_ts{args.timestep}.input_str:previous.start_time:zero.normalizer'
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)
    
    # Load datasets
    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)
    
    # Create data loaders
    train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, 
                                             cxr_val_ds, ehr_test_ds, cxr_test_ds)
    
    # Print data statistics
    print(f"Train samples: {len(train_dl.dataset)}")
    print(f"Validation samples: {len(val_dl.dataset)}")
    print(f"Test samples: {len(test_dl.dataset)}")
    
    # Select appropriate model and trainer based on arguments
    if args.fusion_module == 'attention':
        trainer = AttentionFusionTrainer(train_dl, val_dl, args, test_dl=test_dl)
    elif args.fusion_module == 'transformer':
        trainer = TransformerTrainer(train_dl, val_dl, args, test_dl=test_dl)
    elif args.fusion_module == 'mmtm':
        trainer = MMTMTrainer(train_dl, val_dl, args, test_dl=test_dl)
    elif args.fusion_module == 'daft':
        trainer = DAFTTrainer(train_dl, val_dl, args, test_dl=test_dl)
    else:  # Default MedFuse
        trainer = FusionTrainer(train_dl, val_dl, args, test_dl=test_dl)
    
    # Run training or evaluation
    if args.mode == 'train':
        print("==> Training model...")
        trainer.train()
    elif args.mode == 'eval':
        print("==> Evaluating model...")
        trainer.eval()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()