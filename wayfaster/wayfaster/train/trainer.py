import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .utils import path_to_map
from models.traversability_net import TravNet

class TrainingModule(pl.LightningModule):
    """
    Training module for the TravNet model using PyTorch Lightning.
    """
    def __init__(self, configs):
        """
        Initialize the TrainingModule.

        Args:
            configs (object): Configuration object containing training parameters.
        """
        super().__init__()
        # Save hyperparamters to hparams.yaml
        self.save_hyperparameters()

        # Get training params
        self.eps = 1e-6
        gamma = torch.tensor([configs.TRAINING.GAMMA**i for i in range(configs.TRAINING.HORIZON)])
        self.gamma = nn.Parameter(gamma, requires_grad=False)
        self.learning_rate = configs.OPTIMIZER.LR
        self.weight_decay = configs.OPTIMIZER.WEIGHT_DECAY
        self.depth_weight = configs.TRAINING.DEPTH_WEIGHT

        # Depth training
        self.train_depth = configs.MODEL.TRAIN_DEPTH

        # Depth prediction
        self.predict_depth = configs.MODEL.PREDICT_DEPTH

        # Mapping Model
        self.model = TravNet(
            configs.MODEL.GRID_BOUNDS,
            configs.MODEL.INPUT_SIZE,
            downsample=configs.MODEL.DOWNSAMPLE,
            image_dim=configs.MODEL.LATENT_DIM,
            temporal_length=configs.MODEL.TIME_LENGTH,
            predict_depth=configs.MODEL.PREDICT_DEPTH,
            fuse_pcloud=configs.MODEL.FUSE_PCLOUD
        )

        # Set grid bounds
        self.grid_bounds = configs.MODEL.GRID_BOUNDS
        self.map_size = (
            int((self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0])/self.grid_bounds['ybound'][2]))

        self.map_origin = (
            int((self.grid_bounds['xbound'][1])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1])/self.grid_bounds['ybound'][2]))

        self.map_resolution = (
            self.grid_bounds['xbound'][2],
            self.grid_bounds['ybound'][2])

        # Visualization rate
        self.vis_interval = configs.TRAINING.VIS_INTERVAL

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): Batch of data containing images, point clouds, intrinsics, extrinsics, paths, targets, and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        # Get data
        color_img, pcloud, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask = batch

        # Forward pass
        trav_map, pred_depth, debug = self.model(color_img, pcloud, inv_intrinsics, extrinsics, depth_target)

        # Project path to map
        executed_path = path_to_map(path.unsqueeze(1), torch.ones_like(path[...,0,0]).unsqueeze(1), self.map_size, self.map_resolution, self.map_origin)

        # Calculate traversability loss
        trav_loss, _ = self.trav_criterion(path, trav_map, target_trav, trav_weights)

        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)

        if self.train_depth:
            loss = trav_loss + self.depth_weight*depth_loss
        else:
            loss = trav_loss

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='train')

        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_trav_loss", trav_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_depth_loss", depth_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Batch of data containing images, point clouds, intrinsics, extrinsics, paths, targets, and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        # Get data
        color_img, pcloud, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask = batch

        # Forward pass
        trav_map, pred_depth, debug = self.model(color_img, pcloud, inv_intrinsics, extrinsics, depth_target)

        # Project path to map
        executed_path = path_to_map(path.unsqueeze(1), torch.ones_like(path[...,0,0]).unsqueeze(1), self.map_size, self.map_resolution, self.map_origin)

        # Calculate traversability loss
        trav_loss, trav_error = self.trav_criterion(path, trav_map, target_trav, trav_weights)

        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)

        if self.train_depth:
            loss = trav_loss + self.depth_weight*depth_loss
        else:
            loss = trav_loss

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, executed_path, prefix='valid')

        # Logging to TensorBoard by default
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid_trav_loss", trav_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid_trav_error", trav_error, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid_depth_loss", depth_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            list: List containing the optimizer and the scheduler.
        """
        model_params = self.model.parameters()
        optimizer = torch.optim.AdamW(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

        return [optimizer], [scheduler]

    def trav_criterion(self, states, trav_map, target_trav, target_weights):
        """
        Calculate the traversability loss.

        Args:
            states (torch.Tensor): Tensor containing the states.
            trav_map (torch.Tensor): Tensor containing the traversability map.
            target_trav (torch.Tensor): Tensor containing the target traversability.
            target_weights (torch.Tensor): Tensor containing the target weights.

        Returns:
            tuple: Tuple containing the loss and the error.
        """
        # Calculate traversability
        idxu = 2 * (states[...,0] - self.grid_bounds['xbound'][0]) / (self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0]) - 1
        idxv = 2 * (states[...,1] - self.grid_bounds['ybound'][0]) / (self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0]) - 1
        grid = torch.stack((-idxv, -idxu), -1).unsqueeze(1)
        mask = (grid[...,0] >= -1) * (grid[...,0] <= 1) * (grid[...,1] >= -1) * (grid[...,1] <= 1)

        # Sample from map
        traversability = F.grid_sample(trav_map, grid, align_corners=True)
        traversability = traversability.permute((0,2,3,1)).squeeze(1)

        # Calculate loss
        gamma = self.gamma[:traversability.shape[1]].view(1,-1,1).expand(traversability.shape[0],-1,2)
        error = F.smooth_l1_loss(traversability, target_trav, reduction='none', beta=0.1)
        loss = error * target_weights * gamma
        loss = torch.mean(loss[mask.squeeze(1),:])

        # Calculate error
        error = torch.mean(error[mask.squeeze(1),:])

        return loss, error
    def depth_criterion(self, prediction, target, mask):
        """
        Calculate the depth classification loss.

        Args:
            prediction (torch.Tensor): Tensor containing the predicted depth.
            target (torch.Tensor): Tensor containing the target depth.
            mask (torch.Tensor): Tensor containing the mask.

        Returns:
            torch.Tensor: Depth loss.
        """
        loss = F.cross_entropy(prediction, target, reduction='none') * mask
        return torch.mean(loss)

    def visualize(
        self, image, pcloud, trav_map, pred_depth,
        depth_target, depth_mask, debug, executed_path,
        prefix='train'
    ):
        def log_batch(name, tensor):
            """
            Log up to four individual images from a batch.

            Input:  [N, C, H, W]
            Output: list of [C, H, W] tensors
            """
            if tensor.ndim != 4:
                raise ValueError(
                    f"Expected 4D image batch, got {tensor.shape}"
                )

            images = [
                img.detach().float().cpu()
                for img in tensor[:4]
            ]

            self.logger.log_image(
                key=f"{prefix}_{name}",
                images=images
            )

        # RGB images: [B, T, 3, H, W] -> [B*T, 3, H, W]
        images = image.reshape(-1, *image.shape[2:])
        log_batch("images", images)

        # Point clouds
        pcloud = torch.mean(pcloud, dim=2, keepdim=True)
        pcloud = pcloud.reshape(-1, *pcloud.shape[2:])
        log_batch("pcloud", pcloud)

        # Traversability predictions
        log_batch("mu", trav_map[:, :1])
        log_batch("nu", trav_map[:, 1:])

        # Depth prediction
        n_d = (
            self.grid_bounds['dbound'][1]
            - self.grid_bounds['dbound'][0]
        ) / self.grid_bounds['dbound'][2]

        depth_pred = torch.argmax(
            pred_depth, dim=1, keepdim=True
        ) / (n_d - 1)

        log_batch("depth_pred", depth_pred)

        # Depth target
        if self.predict_depth:
            depth_target = depth_target.unsqueeze(1) / (n_d - 1)
        else:
            depth_target = (
                depth_target.argmax(1).unsqueeze(1) / (n_d - 1)
            )

        log_batch("depth_target", depth_target)

        # Depth mask
        log_batch("depth_mask", depth_mask.unsqueeze(1))

        # Debug output
        temp = torch.sum(debug, dim=1, keepdim=True)
        temp_min = temp.min()
        temp_range = (temp.max() - temp_min).clamp_min(self.eps)
        temp = (temp - temp_min) / temp_range

        log_batch("debug", temp)

        # Executed path
        executed_path = executed_path / (
            torch.amax(
                executed_path, (1, 2, 3), keepdim=True
            ) + self.eps
        )

        log_batch("path", executed_path)
