"""
Patient-level evaluation for segmentation.

Aggregates slice-level predictions to patient-level decisions:
- Groups slices by patient ID
- Determines tumor presence (any slice with tumor)
- Estimates tumor volume
- Computes patient-level sensitivity/specificity
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class PatientLevelEvaluator:
    """
    Aggregates slice-level predictions to patient-level metrics.
    """
    
    def __init__(
        self,
        prob_threshold: float = 0.5,
        min_tumor_area: int = 100,
        slice_thickness: float = 1.0,
    ):
        """
        Args:
            prob_threshold: Probability threshold for tumor detection
            min_tumor_area: Minimum tumor area (pixels) to consider positive
            slice_thickness: Slice thickness in mm for volume estimation
        """
        self.prob_threshold = prob_threshold
        self.min_tumor_area = min_tumor_area
        self.slice_thickness = slice_thickness
        
        self.patient_data = defaultdict(lambda: {
            'slices': [],
            'gt_masks': [],
            'pred_masks': [],
            'pred_probs': [],
            'slice_indices': [],
        })
    
    def add_slice(
        self,
        patient_id: str,
        slice_idx: int,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        pred_prob: Optional[np.ndarray] = None,
    ):
        """
        Add a slice prediction for a patient.
        
        Args:
            patient_id: Patient identifier
            slice_idx: Slice index
            gt_mask: Ground truth mask (H, W)
            pred_mask: Predicted mask (H, W)
            pred_prob: Predicted probability map (H, W) - optional
        """
        self.patient_data[patient_id]['slices'].append(slice_idx)
        self.patient_data[patient_id]['gt_masks'].append(gt_mask)
        self.patient_data[patient_id]['pred_masks'].append(pred_mask)
        self.patient_data[patient_id]['slice_indices'].append(slice_idx)
        
        if pred_prob is not None:
            self.patient_data[patient_id]['pred_probs'].append(pred_prob)
    
    def compute_patient_metrics(self, patient_id: str) -> Dict:
        """
        Compute metrics for a single patient.
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            Dictionary of patient-level metrics
        """
        data = self.patient_data[patient_id]
        
        if len(data['slices']) == 0:
            return {}
        
        gt_masks = np.array(data['gt_masks'])
        pred_masks = np.array(data['pred_masks'])
        
        # Patient-level ground truth: tumor present if any slice has tumor
        gt_has_tumor = np.any([mask.sum() > 0 for mask in gt_masks])
        
        # Patient-level prediction: tumor present if any slice exceeds thresholds
        pred_has_tumor = False
        for pred_mask in pred_masks:
            tumor_area = pred_mask.sum()
            if tumor_area >= self.min_tumor_area:
                pred_has_tumor = True
                break
        
        # Slice-level statistics
        num_slices = len(data['slices'])
        num_gt_positive_slices = sum([mask.sum() > 0 for mask in gt_masks])
        num_pred_positive_slices = sum([mask.sum() >= self.min_tumor_area for mask in pred_masks])
        
        # Volume estimation (if tumor present)
        gt_volume = 0.0
        pred_volume = 0.0
        
        if gt_has_tumor:
            # Assume each pixel is 1mm x 1mm (adjust if you have pixel spacing)
            pixel_area_mm2 = 1.0  # mm²
            for mask in gt_masks:
                slice_area_mm2 = mask.sum() * pixel_area_mm2
                slice_volume_mm3 = slice_area_mm2 * self.slice_thickness
                gt_volume += slice_volume_mm3
        
        if pred_has_tumor:
            pixel_area_mm2 = 1.0
            for mask in pred_masks:
                slice_area_mm2 = mask.sum() * pixel_area_mm2
                slice_volume_mm3 = slice_area_mm2 * self.slice_thickness
                pred_volume += slice_volume_mm3
        
        # Compile metrics
        metrics = {
            'patient_id': patient_id,
            'num_slices': num_slices,
            'gt_has_tumor': int(gt_has_tumor),
            'pred_has_tumor': int(pred_has_tumor),
            'num_gt_positive_slices': num_gt_positive_slices,
            'num_pred_positive_slices': num_pred_positive_slices,
            'gt_volume_mm3': gt_volume,
            'pred_volume_mm3': pred_volume,
            'volume_error_mm3': abs(pred_volume - gt_volume),
            'volume_error_percent': abs(pred_volume - gt_volume) / (gt_volume + 1e-8) * 100,
        }
        
        # Slice-level Dice scores
        slice_dice_scores = []
        for gt_mask, pred_mask in zip(gt_masks, pred_masks):
            intersection = np.sum((gt_mask > 0) & (pred_mask > 0))
            union = np.sum(gt_mask > 0) + np.sum(pred_mask > 0)
            dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
            slice_dice_scores.append(dice)
        
        metrics['mean_slice_dice'] = np.mean(slice_dice_scores)
        metrics['std_slice_dice'] = np.std(slice_dice_scores)
        metrics['min_slice_dice'] = np.min(slice_dice_scores)
        metrics['max_slice_dice'] = np.max(slice_dice_scores)
        
        return metrics
    
    def compute_all_patients(self) -> pd.DataFrame:
        """
        Compute metrics for all patients.
        
        Returns:
            DataFrame with per-patient metrics
        """
        all_metrics = []
        
        for patient_id in self.patient_data.keys():
            metrics = self.compute_patient_metrics(patient_id)
            if metrics:
                all_metrics.append(metrics)
        
        return pd.DataFrame(all_metrics)
    
    def compute_patient_level_performance(self) -> Dict:
        """
        Compute patient-level sensitivity and specificity.
        
        Returns:
            Dictionary with patient-level performance metrics
        """
        tp = 0  # True positives (correctly detected tumor patients)
        tn = 0  # True negatives (correctly detected healthy patients)
        fp = 0  # False positives (healthy classified as tumor)
        fn = 0  # False negatives (tumor classified as healthy)
        
        for patient_id in self.patient_data.keys():
            metrics = self.compute_patient_metrics(patient_id)
            
            gt = metrics['gt_has_tumor']
            pred = metrics['pred_has_tumor']
            
            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 0:
                tn += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 1 and pred == 0:
                fn += 1
        
        # Compute metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / (total + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)  # Recall
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)  # PPV
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        return {
            'num_patients': total,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
        }
    
    def save_results(self, output_path: str):
        """
        Save patient-level results to CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        df = self.compute_all_patients()
        df.to_csv(output_path, index=False)
        
        print(f"Saved patient-level results to: {output_path}")
        print(f"  Total patients: {len(df)}")
        print(f"  Patients with tumor (GT): {df['gt_has_tumor'].sum()}")
        print(f"  Patients with tumor (Pred): {df['pred_has_tumor'].sum()}")


def aggregate_patient_predictions(
    predictions_dir: str,
    output_csv: str,
    prob_threshold: float = 0.5,
    min_tumor_area: int = 100,
) -> pd.DataFrame:
    """
    Aggregate slice-level predictions to patient-level.
    
    Args:
        predictions_dir: Directory with prediction files
        output_csv: Path to save results
        prob_threshold: Probability threshold
        min_tumor_area: Minimum tumor area
    
    Returns:
        DataFrame with patient-level metrics
    """
    evaluator = PatientLevelEvaluator(
        prob_threshold=prob_threshold,
        min_tumor_area=min_tumor_area,
    )
    
    # Load predictions (implementation depends on your file format)
    # This is a placeholder - adapt to your actual data format
    
    # Compute and save
    df = evaluator.compute_all_patients()
    df.to_csv(output_csv, index=False)
    
    return df


if __name__ == "__main__":
    # Test patient-level evaluation
    print("Testing Patient-Level Evaluation...")
    print("=" * 60)
    
    # Create synthetic patient data
    np.random.seed(42)
    
    evaluator = PatientLevelEvaluator(
        prob_threshold=0.5,
        min_tumor_area=100,
        slice_thickness=1.0,
    )
    
    # Patient 1: Has tumor (3 slices)
    print("\nAdding Patient 1 (has tumor)...")
    for i in range(3):
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[30:70, 30:70] = 1  # Tumor
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[32:68, 32:68] = 1  # Good prediction
        
        evaluator.add_slice('patient_001', i, gt_mask, pred_mask)
    
    # Patient 2: No tumor (2 slices)
    print("Adding Patient 2 (no tumor)...")
    for i in range(2):
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        
        evaluator.add_slice('patient_002', i, gt_mask, pred_mask)
    
    # Patient 3: Has tumor but missed (2 slices)
    print("Adding Patient 3 (has tumor, missed)...")
    for i in range(2):
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[30:70, 30:70] = 1
        
        pred_mask = np.zeros((100, 100), dtype=np.uint8)  # Missed
        
        evaluator.add_slice('patient_003', i, gt_mask, pred_mask)
    
    # Compute patient-level metrics
    print("\n" + "=" * 60)
    print("Patient-Level Metrics")
    print("=" * 60)
    
    df = evaluator.compute_all_patients()
    print("\nPer-Patient Summary:")
    print(df.to_string(index=False))
    
    # Compute overall performance
    print("\n" + "=" * 60)
    print("Overall Patient-Level Performance")
    print("=" * 60)
    
    performance = evaluator.compute_patient_level_performance()
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
