import AnomalyCLIP_lib
import torch
import argparse
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from dataset_adapters import ensure_dataset_ready
from logger import get_logger
from tqdm import tqdm

import json
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer
from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
def test(args):
    img_size = args.image_size
    features_list = args.features_list
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root, meta_path = ensure_dataset_ready(dataset_name, args.data_path)
    logger.info("Using dataset root: %s", dataset_root)
    logger.info("Using meta file: %s", meta_path)

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(
        root=dataset_root,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list


    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)


    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader, desc="Testing")):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                    # The following code is equivalent. 
                    # anomaly_map = similarity_map[...,1] 
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            
            anomaly_map = anomaly_map.sum(dim = 0)
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            # visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)

    logger.info("Starting metrics calculation for each object...")
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    per_object_metrics = {}
    for obj in obj_list:
        logger.info(f"Processing metrics for object: {obj}")
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            logger.info(f"Calculating image-level metrics for {obj}")
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            per_object_metrics[obj] = {
                "image_auroc": float(image_auroc),
                "image_ap": float(image_ap),
            }
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            logger.info(f"Image-level metrics for {obj} - AUROC: {image_auroc:.3f}, AP: {image_ap:.3f}")
        elif args.metrics == 'pixel-level':
            logger.info(f"Calculating pixel-level metrics for {obj}")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            per_object_metrics[obj] = {
                "pixel_auroc": float(pixel_auroc),
                "pixel_aupro": float(pixel_aupro),
            }
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            logger.info(f"Pixel-level metrics for {obj} - AUROC: {pixel_auroc:.3f}, AUPRO: {pixel_aupro:.3f}")
        elif args.metrics == 'image-pixel-level':
            logger.info(f"Calculating both image and pixel-level metrics for {obj}")
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            per_object_metrics[obj] = {
                "image_auroc": float(image_auroc),
                "image_ap": float(image_ap),
                "pixel_auroc": float(pixel_auroc),
                "pixel_aupro": float(pixel_aupro),
            }
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            logger.info(f"Combined metrics for {obj} - Image AUROC: {image_auroc:.3f}, Image AP: {image_ap:.3f}, Pixel AUROC: {pixel_auroc:.3f}, Pixel AUPRO: {pixel_aupro:.3f}")
        table_ls.append(table)

    logger.info("Calculating mean metrics across all objects...")
    if args.metrics == 'image-level':
        mean_image_auroc = np.mean(image_auroc_list) * 100
        mean_image_ap = np.mean(image_ap_list) * 100
        table_ls.append(['mean', 
                        str(np.round(mean_image_auroc, decimals=1)),
                        str(np.round(mean_image_ap, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
        logger.info(f"Mean image-level metrics - AUROC: {mean_image_auroc:.1f}%, AP: {mean_image_ap:.1f}%")
    elif args.metrics == 'pixel-level':
        mean_pixel_auroc = np.mean(pixel_auroc_list) * 100
        mean_pixel_aupro = np.mean(pixel_aupro_list) * 100
        table_ls.append(['mean', str(np.round(mean_pixel_auroc, decimals=1)),
                        str(np.round(mean_pixel_aupro, decimals=1))
                       ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
        logger.info(f"Mean pixel-level metrics - AUROC: {mean_pixel_auroc:.1f}%, AUPRO: {mean_pixel_aupro:.1f}%")
    elif args.metrics == 'image-pixel-level':
        mean_pixel_auroc = np.mean(pixel_auroc_list) * 100
        mean_pixel_aupro = np.mean(pixel_aupro_list) * 100
        mean_image_auroc = np.mean(image_auroc_list) * 100
        mean_image_ap = np.mean(image_ap_list) * 100
        table_ls.append(['mean', str(np.round(mean_pixel_auroc, decimals=1)),
                        str(np.round(mean_pixel_aupro, decimals=1)), 
                        str(np.round(mean_image_auroc, decimals=1)),
                        str(np.round(mean_image_ap, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
        logger.info(f"Mean combined metrics - Image AUROC: {mean_image_auroc:.1f}%, Image AP: {mean_image_ap:.1f}%, Pixel AUROC: {mean_pixel_auroc:.1f}%, Pixel AUPRO: {mean_pixel_aupro:.1f}%")
    summary = {
        "method": args.method_name,
        "dataset": dataset_name,
        "data_path": dataset_root,
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "metrics_mode": args.metrics,
        "per_object": per_object_metrics,
    }

    if args.metrics == 'image-level':
        summary["metrics"] = {
            "I-AUROC": float(np.mean(image_auroc_list)),
            "I-AP": float(np.mean(image_ap_list)),
        }
    elif args.metrics == 'pixel-level':
        summary["metrics"] = {
            "P-AUROC": float(np.mean(pixel_auroc_list)),
            "AU-PRO": float(np.mean(pixel_aupro_list)),
        }
    elif args.metrics == 'image-pixel-level':
        summary["metrics"] = {
            "I-AUROC": float(np.mean(image_auroc_list)),
            "P-AUROC": float(np.mean(pixel_auroc_list)),
            "AU-PRO": float(np.mean(pixel_aupro_list)),
            "I-AP": float(np.mean(image_ap_list)),
        }

    logger.info("\nFinal results table:")
    logger.info("\n%s", results)

    if args.results_json:
        results_json_path = os.path.abspath(args.results_json)
        os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
        with open(results_json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        logger.info("Saved JSON summary to %s", results_json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--results_json", type=str, default="")
    parser.add_argument("--method_name", type=str, default="anomalyclip")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
