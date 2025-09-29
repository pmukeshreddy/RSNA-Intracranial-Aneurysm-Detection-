import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class EnsemblePredictor:
    """Ensemble predictor combining ResNet, EfficientNet, and ViT"""
    
    def __init__(self, model_paths, weights=None, device='cuda'):
        """
        Args:
            model_paths: dict with keys 'resnet', 'efficientnet', 'vit' and checkpoint paths
            weights: Optional dict with model weights for weighted averaging
            device: Device to run inference on
        """
        self.device = device
        self.models = {}
        
        # Default equal weights if not specified
        if weights is None:
            weights = {'resnet': 1.0, 'efficientnet': 1.0, 'vit': 1.0}
        self.weights = weights
        
        print("Loading ensemble models...")
        
        # Load ResNet
        if 'resnet' in model_paths:
            print("Loading ResNet50...")
            self.models['resnet'] = ResNet3D(in_channels=3, num_classes=14, dropout=0.5)
            checkpoint = torch.load(model_paths['resnet'], map_location=device)
            if 'model_state_dict' in checkpoint:
                self.models['resnet'].load_state_dict(checkpoint['model_state_dict'])
            else:
                self.models['resnet'].load_state_dict(checkpoint)
            self.models['resnet'].to(device).eval()
        
        # Load EfficientNet
        if 'efficientnet' in model_paths:
            print("Loading EfficientNet-B3...")
            self.models['efficientnet'] = efficientnet_b3_3d(in_channels=3, num_classes=14, dropout=0.5)
            checkpoint = torch.load(model_paths['efficientnet'], map_location=device)
            if 'model_state_dict' in checkpoint:
                self.models['efficientnet'].load_state_dict(checkpoint['model_state_dict'])
            else:
                self.models['efficientnet'].load_state_dict(checkpoint)
            self.models['efficientnet'].to(device).eval()
        
        # Load ViT
        if 'vit' in model_paths:
            print("Loading Vision Transformer...")
            self.models['vit'] = VisionTransformer3D(
                img_size=(64, 128, 128),
                patch_size=(8, 16, 16),
                in_channels=3,
                num_classes=14,
                embed_dim=384,
                depth=6,
                num_heads=6,
                drop_rate=0.3
            )
            checkpoint = torch.load(model_paths['vit'], map_location=device)
            if 'model_state_dict' in checkpoint:
                self.models['vit'].load_state_dict(checkpoint['model_state_dict'])
            else:
                self.models['vit'].load_state_dict(checkpoint)
            self.models['vit'].to(device).eval()
        
        print(f"✅ Loaded {len(self.models)} models for ensemble")
    
    @torch.no_grad()
    def predict(self, dataloader, method='weighted_avg'):
        """
        Run ensemble prediction
        
        Args:
            dataloader: DataLoader for inference
            method: 'weighted_avg', 'avg', 'max', 'voting'
        
        Returns:
            predictions: Numpy array of predictions
            targets: Numpy array of ground truth labels (if available)
        """
        all_predictions = {name: [] for name in self.models.keys()}
        all_targets = []
        
        print(f"\nRunning ensemble inference with method: {method}")
        
        for inputs, targets in tqdm(dataloader, desc="Ensemble prediction"):
            inputs = inputs.to(self.device)
            
            # Get predictions from each model
            for name, model in self.models.items():
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                all_predictions[name].append(probs.cpu())
            
            all_targets.append(targets)
        
        # Concatenate all batches
        for name in all_predictions:
            all_predictions[name] = torch.cat(all_predictions[name], dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Combine predictions
        if method == 'weighted_avg':
            # Weighted average based on model weights
            total_weight = sum(self.weights.values())
            ensemble_pred = np.zeros_like(all_predictions[list(self.models.keys())[0]])
            for name, preds in all_predictions.items():
                ensemble_pred += preds * (self.weights[name] / total_weight)
        
        elif method == 'avg':
            # Simple average
            ensemble_pred = np.mean([preds for preds in all_predictions.values()], axis=0)
        
        elif method == 'max':
            # Take maximum probability
            ensemble_pred = np.maximum.reduce([preds for preds in all_predictions.values()])
        
        elif method == 'voting':
            # Hard voting (threshold at 0.5)
            votes = np.array([preds > 0.5 for preds in all_predictions.values()])
            ensemble_pred = np.mean(votes, axis=0)
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred, all_targets, all_predictions
    
    def evaluate(self, dataloader, method='weighted_avg'):
        """Evaluate ensemble on validation data"""
        predictions, targets, individual_preds = self.predict(dataloader, method)
        
        # Calculate ensemble metrics
        print("\n" + "="*60)
        print("ENSEMBLE RESULTS")
        print("="*60)
        
        # Per-label AUC
        label_aucs = {}
        for i, col in enumerate(LABEL_COLS):
            if targets[:, i].sum() > 0 and (targets[:, i] == 0).sum() > 0:
                label_aucs[col] = roc_auc_score(targets[:, i], predictions[:, i])
        
        main_auc = label_aucs.get('Aneurysm Present', 0.0)
        mean_auc = np.mean(list(label_aucs.values())) if label_aucs else 0.0
        
        # Kaggle metric
        ap = label_aucs.get('Aneurysm Present', 0.0)
        others = [auc for k, auc in label_aucs.items() if k != 'Aneurysm Present']
        avg_others = np.mean(others) if len(others) > 0 else 0.0
        kaggle_score = 0.5 * (ap + avg_others)
        
        print(f"\nEnsemble ({method}):")
        print(f"  Main AUC (Aneurysm Present): {main_auc:.4f}")
        print(f"  Mean AUC (all labels): {mean_auc:.4f}")
        print(f"  Kaggle Score: {kaggle_score:.4f}")
        
        # Individual model performance
        print("\nIndividual Model Performance:")
        for name, preds in individual_preds.items():
            model_aucs = {}
            for i, col in enumerate(LABEL_COLS):
                if targets[:, i].sum() > 0 and (targets[:, i] == 0).sum() > 0:
                    model_aucs[col] = roc_auc_score(targets[:, i], preds[:, i])
            
            model_kaggle = 0.5 * (
                model_aucs.get('Aneurysm Present', 0.0) + 
                np.mean([auc for k, auc in model_aucs.items() if k != 'Aneurysm Present'])
            )
            print(f"  {name.capitalize()}: {model_kaggle:.4f}")
        
        print("\nPer-label AUC (Ensemble):")
        for label, auc in label_aucs.items():
            print(f"  {label}: {auc:.4f}")
        
        return {
            'ensemble_kaggle': kaggle_score,
            'ensemble_main_auc': main_auc,
            'ensemble_mean_auc': mean_auc,
            'label_aucs': label_aucs,
            'individual_scores': {name: preds for name, preds in individual_preds.items()}
        }


# Usage Example
if __name__ == "__main__":
    # Define paths to trained models
    model_paths = {
        'resnet': './checkpoints_ensemble_resnet/model_fold0_best.pth',
        'efficientnet': './checkpoints_ensemble_efficientnet/model_fold1_best.pth',
        'vit': './checkpoints_ensemble_vit/model_fold2_best.pth'
    }
    
    # Optional: Set custom weights for each model
    # (based on their individual performance)
    model_weights = {
        'resnet': 1.0,
        'efficientnet': 1.2,  # Slightly higher weight if it performs better
        'vit': 1.0
    }
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(
        model_paths=model_paths,
        weights=model_weights,
        device='cuda'
    )
    
    # Create validation data module (use fold not used in training)
    val_data_module = AneurysmDataModule(
        npz_dir='/content/drive/MyDrive/kaggle_sep_2025_main/New Folder With Items 2',
        train_csv='/content/drive/MyDrive/train.csv',
        batch_size=16,
        num_workers=8,
        fold=3,  # Use a fold not used in training
        use_2d=False,
        cache_data=False
    )
    
    val_loader = val_data_module.val_dataloader()
    
    # Evaluate with different ensemble methods
    print("\nTrying different ensemble methods...\n")
    
    methods = ['weighted_avg', 'avg', 'max']
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print('='*60)
        result = ensemble.evaluate(val_loader, method=method)
        results[method] = result
    
    # Find best method
    best_method = max(results.items(), key=lambda x: x[1]['ensemble_kaggle'])
    print(f"\n{'='*60}")
    print(f"BEST ENSEMBLE METHOD: {best_method[0].upper()}")
    print(f"Kaggle Score: {best_method[1]['ensemble_kaggle']:.4f}")
    print('='*60)
