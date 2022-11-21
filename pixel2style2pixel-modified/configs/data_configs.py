from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    "ffhq_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ffhq_512"],
        "train_target_root": dataset_paths["ffhq_512"],
        "test_source_root": dataset_paths["ffhq_512_val"],
        "test_target_root": dataset_paths["ffhq_512_val"],
    },
    "ffhq_encode_cond": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ffhq_512_cond"],
        "train_target_root": dataset_paths["ffhq_512_cond"],
        "test_source_root": dataset_paths["ffhq_512_cond_val"],
        "test_target_root": dataset_paths["ffhq_512_cond_val"],
        "train_labels": dataset_paths["ffhq_512_train_labels"],
        "test_labels": dataset_paths["ffhq_512_test_labels"]
    },
    "paired_gens": {
        "transforms": transforms_config.PairedEncodeTransforms,
        "train_source_root": dataset_paths["paired_gens_input"],
        "train_target_root": dataset_paths["paired_gens_output"],
        "test_source_root": dataset_paths["paired_gens_val_input"],
        "test_target_root": dataset_paths["paired_gens_val_output"],
    },
    "paired_gens_latent": {
        "transforms": transforms_config.PairedEncodeTransforms,
        "train_source_root": dataset_paths["paired_gens_input"],
        "train_target_root": dataset_paths["paired_gens_output"],
        "train_latents_root": dataset_paths["paired_gens_latents"],
        "test_source_root": dataset_paths["paired_gens_val_input"],
        "test_target_root": dataset_paths["paired_gens_val_output"],
        "test_latents_root": dataset_paths["paired_gens_val_latents"],
    },
    "ffhq_frontalize": {
        "transforms": transforms_config.FrontalizationTransforms,
        "train_source_root": dataset_paths["ffhq"],
        "train_target_root": dataset_paths["ffhq"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_sketch_to_face": {
        "transforms": transforms_config.SketchToImageTransforms,
        "train_source_root": dataset_paths["celeba_train_sketch"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test_sketch"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_seg_to_face": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": dataset_paths["celeba_train_segmentation"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test_segmentation"],
        "test_target_root": dataset_paths["celeba_test"],
    },
    "celebs_super_resolution": {
        "transforms": transforms_config.SuperResTransforms,
        "train_source_root": dataset_paths["celeba_train"],
        "train_target_root": dataset_paths["celeba_train"],
        "test_source_root": dataset_paths["celeba_test"],
        "test_target_root": dataset_paths["celeba_test"],
    },
}
