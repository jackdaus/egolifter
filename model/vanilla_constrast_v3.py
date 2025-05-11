# Based on Meta's EgoLifter. See base repo for license details.

import numpy as np
import torch
import wandb

from scene.gaussian_mlp_model import GaussianMLPModel
from utils.vis import concate_images_vis, feat_image_to_pca_vis

from .vanilla import VanillaGaussian


class VanillaContrastV3(VanillaGaussian):
    """
    Learning with a mlp learning the lifted segmentation.
    """

    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)

        print("Initializing VanillaContrastV3...")

        if self.cfg.model.dim_extra == 0:
            raise ValueError(
                "dim_extra must be greater than 0 in order to use the GaussianMLPModel!"
            )

        # TODO confirm if it's okay to set the self.gaussians when the parent class has already set it. Does the old gradient stuff linger around?
        self.gaussians = GaussianMLPModel(
            self.cfg.model.sh_degree,
            self.cfg.model.dim_extra,
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        self.log_extra_features_vis()

    def on_train_epoch_end(self):
        self.log_extra_features_vis()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Log images on the first training batch and every 1000 steps (only on rank‑0).
        """
        if self.global_step == 0:
            self.log_image_train(batch, outputs)
        elif self.global_step % 300 == 0 and self.trainer.is_global_zero:
            self.log_image_train(batch, outputs)

    def log_extra_features_vis(self):
        # Log the extra features as a point cloud in wandb.
        if self.gaussians.dim_extra == 3:
            # Log the extra feature dims as a point cloud in wandb.
            extra_features_3D = self.gaussians.get_features_extra.detach().cpu().numpy()
            point_scene = wandb.Object3D(
                {"type": "lidar/beta", "points": extra_features_3D}
            )
            wandb.log({"gaussian_vis/extra_features_3d": point_scene}, commit=True)
        # TODO check if other dim visuals work as expected.
        elif self.gaussians.dim_extra == 2:
            extra_features_2D = self.gaussians.get_features_extra.detach().cpu().numpy()
            # use wandb scatter plot
            # extra_features_2D = extra_features_2D.reshape(-1, 2)
            table = wandb.Table(data=extra_features_2D, columns=["x", "y"])
            wandb.log({"gaussian_vis/extra_features_2d": table}, commit=True)
        elif self.gaussians.dim_extra == 1:
            # Log the extra feature dims as a 1D image in wandb.
            extra_features_1D = self.gaussians.get_features_extra.detach().cpu().numpy()
            # log in wandb as a histogram
            wandb.log(
                {
                    "gaussian_vis/init_extra_features_1d": wandb.Histogram(
                        extra_features_1D
                    )
                },
                commit=True,
            )
        elif self.gaussians.dim_extra > 3:
            # Use PCA or tSNE to reduce the dimensionality of the extra features for 3D visualization.
            # TODO
            pass

    def log_image_train(self, batch, outputs):
        with torch.no_grad():
            image_rendered = outputs["render_pkg"]["render"].clamp(0.0, 1.0)
            # image_processed = outputs["image_processed"].clamp(0.0, 1.0)
            # gt_image_processed = outputs["gt_image_processed"].clamp(0.0, 1.0)
            render_pkg_feat = outputs["render_pkg_feat"]

            subset = batch["subset"][0]

            viewpoint_cam = self.scene.get_camera(batch["idx"].item(), subset=subset)
            image_name = viewpoint_cam.image_name
            log_name = subset + "_view/{}".format(image_name)

            render_wandb = (
                image_rendered.clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
                * 255
            ).astype(np.uint8)

            # image_wandb = (
            #     image_processed.clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
            #     * 255
            # ).astype(np.uint8)

            # gt_image_wandb = (
            #     gt_image_processed.clamp(0.0, 1.0)
            #     .permute(1, 2, 0)
            #     .contiguous()
            #     .cpu()
            #     .numpy()
            #     * 255
            # ).astype(np.uint8)

            # Stack images horizontally
            images_wandb = [render_wandb]
            caption = "Render (direct)"

            if render_pkg_feat is not None:
                feat_image = render_pkg_feat["render_features"].detach()  # (D, H, W)
                feat_image = feat_image_to_pca_vis(feat_image, channel_first=True)
                feat_image_wandb = (feat_image * 255).astype(np.uint8)
                images_wandb.append(feat_image_wandb)

            image_wandb = concate_images_vis(images_wandb)
            image_wandb = wandb.Image(image_wandb, caption=caption)
            wandb.log({log_name: image_wandb}, commit=True)
