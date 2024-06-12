import multiprocessing
import monai
from skimage import morphology
import numpy as np
import logging
import os
import argparse
from dataclasses import dataclass
import pickle

# LEFT-RIGHT CONFUSION

def kidney_adrenal_left_right_confusion(predicted_label_map: np.ndarray):
    """
     Expects predicted_label_map with dimensions: z y x
    """
    shape = predicted_label_map.shape
    mid_id_z, mid_id_y, mid_id_x = shape[0] / 2, shape[1] / 2, shape[2] / 2
    kidney_labels = {"right": 2, "left": 3}
    adrenal_labels = {"right": 11, "left": 12}
    
    all_bin_maps = []
    all_organ_maps = []
    for organ_label in [kidney_labels, adrenal_labels]:
        bin_organ_map = (predicted_label_map == organ_label["right"]) | (predicted_label_map == organ_label["left"])
        dilate_organs = morphology.dilation(bin_organ_map, morphology.ball(radius=3))
        # Kidneys tend to be far apart -> maybe bigger rad?
        labeled_organs = morphology.label(dilate_organs, connectivity=3)
        labeled_organs[~bin_organ_map] = 0.  # Undo the dilation
        
        foreground, foreground_cnt = np.unique(labeled_organs, return_counts=True)
        foreground = foreground[1:]  # Removing background label
        foreground_cnt = foreground_cnt[1:]  # Removing background label
    
        assigned_labels = {"right": [], "left": []}
        for comp_id, counts in zip(foreground, foreground_cnt):  # Skip 0 by starting at next val
            ids = np.argwhere(labeled_organs == comp_id)
            avg_id_z, avg_id_y, avg_id_x = list(np.mean(ids, axis=0))
            if avg_id_x > mid_id_x:
                assigned_labels["right"].append(comp_id)
            else:
                assigned_labels["left"].append(comp_id)
                
        if len(assigned_labels["left"]) != 0:
            left_organ_bin_map = np.sum(np.stack([labeled_organs == i for i in assigned_labels["left"]], axis=0), axis=0)
        else:
            left_organ_bin_map = np.zeros_like(labeled_organs)
            
        if len(assigned_labels["right"]) != 0:
            right_organ_bin_map = np.sum(np.stack([labeled_organs == i for i in assigned_labels["right"]], axis=0), axis=0)
        else:
            right_organ_bin_map = np.zeros_like(labeled_organs)
            
        left_organ_map = left_organ_bin_map * organ_label["left"]
        right_organ_map = right_organ_bin_map * organ_label["right"]
        all_organ_maps.append(left_organ_map.copy())
        all_organ_maps.append(right_organ_map.copy())
        all_bin_maps.append(bin_organ_map.copy())
    
    all_bin_maps = np.sum(np.stack(all_bin_maps, axis=0),axis=0)
    all_organ_maps = np.sum(np.stack(all_organ_maps, axis=0), axis=0)
    
    final_organ_maps = np.where(all_bin_maps, all_organ_maps, predicted_label_map)

    return final_organ_maps

# CONNECTED COMPONENT FILTERING

organs = [
    "Background",
    "Spleen",
    "Right Kidney",
    "Left Kidney",
    "Gallbladder",
    "Esophagus",
    "Liver",
    "Stomach",
    "Aorta",
    "Inferior Vena Cava",
    "Pancreas",
    "Right Adrenal Gland",
    "Left Adrenal Gland",
]

@dataclass
class OrganMap:
    original_organ_id: int
    original_organ_name: str
    organ_voxels: int
    organ_map: np.ndarray
    final_organ_id: int = -1
    final_organ_name: str = ""

def get_connected_organ_maps(predicted_label_map: np.ndarray, organs_of_interest: dict[str, int]):
    """
    Goes through the different organ maps and does a connected component analysis on them.
    All the unconnected regions are saved as an `OrganMap` containing the binary map, the organ value and the name.
    """
    organ_maps: list[OrganMap] = []
    for organ_name, organ_value in organs_of_interest.items():
        bin_org_map = predicted_label_map == organ_value
        labeled_organ_map: np.ndarray = morphology.label(bin_org_map, connectivity=3)
        ids = np.unique(labeled_organ_map)
        for id in ids[1:]:
            largest_volume_bin_map: np.ndarray = labeled_organ_map == id
            organ_maps.append(
                OrganMap(
                    organ_map=largest_volume_bin_map.astype(np.uint8),
                    original_organ_id=organ_value,
                    organ_voxels=int(np.sum(largest_volume_bin_map)),
                    original_organ_name=organ_name,
                )
            )
    return organ_maps

def only_largest_region_filtering(predicted_label_map: np.ndarray):
    organ_maps = get_connected_organ_maps(predicted_label_map, organs_of_interest={c: i+1 for i, c in enumerate(organs[1:])})
    organ_ids = list(set([om.original_organ_id for om in organ_maps]))
    rem_organ_bin_maps = []
    for oid in organ_ids:
        rem_oms: list[OrganMap] = [om for om in organ_maps if om.original_organ_id == oid]
        om_size_of_oms_with_oid = [om.organ_voxels for om in rem_oms]
        max_id = np.argmax(om_size_of_oms_with_oid).astype(int)
        rem_organ_bin_maps.append(rem_oms[max_id].organ_map.astype(bool))
    remaining_foreground = np.sum(np.stack(rem_organ_bin_maps, axis=0), axis=0).astype(bool)
    predicted_label_map[~remaining_foreground] = 0
    return predicted_label_map

# ORGAN SIZE CONSTRAINTS

def small_organ_filtering(predicted_label_map: np.ndarray, spacing, rate: float = 1.):
    assert 0. < rate <= 1., "Rate can only be between 0 and 1"
    vol_per_voxel = float(np.prod(spacing))
    organ_maps = get_connected_organ_maps(predicted_label_map, organs_of_interest={c: i+1 for i, c in enumerate(organs[1:])})
    remaining_organ_maps = [om.organ_map for om in organ_maps if (om.organ_voxels*vol_per_voxel) > (min_gt_volume[str(om.original_organ_id)] * rate)]
    remaining_foreground_bin_map = np.sum(np.stack(remaining_organ_maps, axis=0),axis=0).astype(bool)
    filtered_map = predicted_label_map.copy()
    filtered_map[~remaining_foreground_bin_map] = 0
    return filtered_map

# THIS IS CONNECTED COMPONENT FILTERING + ORGAN SIZE CONSTRAINTS
def small_organ_filtering_but_at_least_one_instance(predicted_label_map: np.ndarray, spacing, rate: float = 1.):
    assert 0. < rate <= 1., "Rate can only be between 0 and 1"
    vol_per_voxel = float(np.prod(spacing))
    organ_maps = get_connected_organ_maps(predicted_label_map, organs_of_interest={c: i+1 for i, c in enumerate(organs[1:])})
    organ_ids = list(set([om.original_organ_id for om in organ_maps]))
    rem_organ_bin_maps = []
    for oid in organ_ids:
        rem_oms: list[OrganMap] = [om for om in organ_maps if om.original_organ_id == oid]
        if len(rem_oms) == 1:
            # If only one no need to check size since its largest
            rem_organ_bin_maps.append(rem_oms[0].organ_map.astype(bool))
        else:
            # Determine the largest
            om_size_of_oms_with_oid = [om.organ_voxels for om in rem_oms]
            max_id = int(np.argmax(om_size_of_oms_with_oid))
            for cnt, om in enumerate(rem_oms):
                if cnt == max_id:
                    # Always add largest
                    rem_organ_bin_maps.append(om.organ_map)
                else:
                    # Maybe add the others if not too small!
                    if (om.organ_voxels * vol_per_voxel) > (min_gt_volume[str(om.original_organ_id)] * rate):
                        rem_organ_bin_maps.append(om.organ_map)
    remaining_foreground = np.sum(np.stack(rem_organ_bin_maps, axis=0), axis=0).astype(bool)
    predicted_label_map[~remaining_foreground] = 0
    return predicted_label_map

def nnunet_dice(mask_pred: np.ndarray, mask_ref: np.ndarray):
    dice_scores = []
    for organ_class in range(1, 13):
        gt = (mask_ref == organ_class)
        pred = (mask_pred == organ_class)
        use_mask = np.ones_like(gt, dtype=bool)
        
        tp = np.sum((gt & pred) & use_mask)
        fp = np.sum(((~gt) & pred) & use_mask)
        fn = np.sum((gt & (~pred)) & use_mask)
        
        if tp + fp + fn == 0:
            dice_scores.append(np.nan)
        else:
            dice_scores.append(2 * tp / (2 * tp + fp + fn))
    
    return dice_scores

def run(data):
    logger = logging.getLogger()
    logger.info('entered')
    transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["pred", "gt"], image_only=False),
        monai.transforms.EnsureChannelFirstd(keys=["pred", "gt"]),
        monai.transforms.CropForegroundd(keys=['pred', 'gt'], source_key='gt', margin=[50, 50, 0]),
    ])

    results = transforms(data)
    pred = results['pred']
    gt = results['gt']

    if args.pp_config != 0:
    
        pred = np.transpose(pred[0], (2, 1, 0))

        pp_pred = kidney_adrenal_left_right_confusion(pred)
        
        if args.pp_config == 3:
            pp_pred = only_largest_region_filtering(pp_pred)

        if args.pp_config == 2:
            pp_pred = small_organ_filtering_but_at_least_one_instance(pp_pred, spacing=results['pred_meta_dict']['pixdim'][1:4], rate=args.rate)

        if args.pp_config == 1:
            pp_pred = small_organ_filtering(pp_pred, spacing=results['pred_meta_dict']['pixdim'][1:4], rate=args.rate)
            
        pp_pred = np.transpose(pp_pred, (2, 1, 0))
        
        dice_score = nnunet_dice(pp_pred, gt)
        logger.info(dice_score)
    
    else:
        dice_score = nnunet_dice(pred, gt)

    return dice_score

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp_config', type=int, required=True)
    parser.add_argument('--rate', type=float)

    args = parser.parse_args()

    with open("/data_analysis/accepted_names", "rb") as fp:   # Unpickling
        accepted_names = pickle.load(fp)

    inference_path = "/nnUNet/inference"
    for fold in range(0, 5):
        print(f'Starting fold {fold}.')
        min_gt_volume = {}
        for key, value in enumerate(np.load(f"/nnUNet_inference/organ_volumes_per_fold/organ_volumes_fold_{fold}.npy")):
            min_gt_volume[str(key+1)] = value
        fold_path = os.path.join(inference_path, 'fold_'+str(fold))
        paths = [{"pred": pred, "gt":gt} for pred, gt in zip(sorted([os.path.join(fold_path, 'preds', f_pred) for f_pred in os.listdir(fold_path + '/preds') if f_pred.endswith('.nii.gz')]), sorted([os.path.join(fold_path, 'gt', f_pred) for f_pred in os.listdir(fold_path + '/gt') if f_pred.endswith('.nii.gz')])) if pred.split('/')[-1].split('.')[0] in accepted_names]
        with multiprocessing.Pool(processes=7) as pool:
            result = pool.map(run, paths)
        if args.pp_config in [1, 2]:
            np.save(f'/nnUNet_inference/post-processing/results/pp{args.pp_config}/rate_{args.rate}/fold_{fold}.npy', np.array(result))
        elif args.pp_config in [3, 4]:
            np.save(f'/nnUNet_inference/post-processing/results/pp{args.pp_config}/fold_{fold}.npy', np.array(result))
        else:
            np.save(f'/nnUNet_inference/post-processing/results/no_pp/fold_{fold}.npy', np.array(result))