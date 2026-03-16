import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import profile_execution
from utils.memory_utils import log_cuda_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S:",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# CNN Architecture.


class NIRPredictionCNN(nn.Module):
    """
    Convolutional Neural Network for NIR prediction from RGB patches.
    Takes 50x50x3 RGB patches and predicts 50x50 NIR values.
    """

    def __init__(self):
        super().__init__()

        # Encoder (downsampling path)
        self.encoder1 = self._conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 50x50 -> 25x25

        self.encoder2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 25x25 -> 12x12

        self.encoder3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 12x12 -> 6x6

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder3 = self._conv_block(512, 256)  # 512 because of skip connection

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder2 = self._conv_block(256, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder1 = self._conv_block(128, 64)

        # Keep predictions non-negative to match the target range.
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),  # Ensure non-negative predictions
        )

    def _conv_block(self, in_channels, out_channels):
        """Convolutional block with two conv layers + ReLU + BatchNorm"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # 50x50x64
        p1 = self.pool1(enc1)  # 25x25x64

        enc2 = self.encoder2(p1)  # 25x25x128
        p2 = self.pool2(enc2)  # 12x12x128

        enc3 = self.encoder3(p2)  # 12x12x256
        p3 = self.pool3(enc3)  # 6x6x256

        # Bottleneck
        bottleneck = self.bottleneck(p3)  # 6x6x512

        up3 = self.up3(bottleneck)  # Upsample: 6x6 -> 12x12
        up3 = self.up_conv3(up3)  # Conv to reduce channels: 512 -> 256
        # Match tensor sizes before concatenation.
        if up3.shape[2:] != enc3.shape[2:]:
            up3 = nn.functional.interpolate(up3, size=enc3.shape[2:], mode="bilinear", align_corners=True)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))  # 12x12x256

        up2 = self.up2(dec3)  # Upsample: 12x12 -> 24x24
        up2 = self.up_conv2(up2)  # Conv to reduce channels: 256 -> 128
        # Align spatial size with enc2 before concatenation.
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = nn.functional.interpolate(up2, size=enc2.shape[2:], mode="bilinear", align_corners=True)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))  # 25x25x128

        up1 = self.up1(dec2)  # Upsample: 25x25 -> 50x50
        up1 = self.up_conv1(up1)  # Conv to reduce channels: 128 -> 64
        # Align spatial size with enc1 before concatenation.
        if up1.shape[2:] != enc1.shape[2:]:
            up1 = nn.functional.interpolate(up1, size=enc1.shape[2:], mode="bilinear", align_corners=True)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))  # 50x50x64

        out = self.out_conv(dec1)  # 50x50x1

        return out


# Dataset for Patch-based Training.


class PatchDataset(Dataset):
    """Dataset that generates patches from RGB images and corresponding NIR values."""

    def __init__(self, patches_rgb, patches_nir):
        """
        Args:
            patches_rgb: numpy array of shape (N, 50, 50, 3)
            patches_nir: numpy array of shape (N, 50, 50)
        """
        self.patches_rgb = patches_rgb
        self.patches_nir = patches_nir

    def __len__(self):
        return len(self.patches_rgb)

    def __getitem__(self, idx):
        rgb = self.patches_rgb[idx]  # (50, 50, 3)
        nir = self.patches_nir[idx]  # (50, 50)

        # Convert to torch tensors and transpose RGB to (3, 50, 50)
        rgb_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        nir_tensor = torch.from_numpy(nir.copy()).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]

        return rgb_tensor, nir_tensor


# CNN Model Wrapper.


class CNNNir:
    """CNN-based NIR prediction model that works with the existing infrastructure."""

    def __init__(
        self,
        patch_size=50,
        batch_size=32,
        epochs=20,
        learning_rate=0.0004,
        validation_split=0.2,
        num_workers=None,
    ):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validation_split = validation_split

        if num_workers is None:
            self.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)) // 2
        else:
            self.num_workers = num_workers

        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")
        log.info(f"PyTorch threads: {torch.get_num_threads()}")
        log.info(f"DataLoader workers: {self.num_workers}")
        log_cuda_info()

        self.model = NIRPredictionCNN().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.patches_rgb = []
        self.patches_nir = []
        self.training_files = set()

        self.timing = {"processing": 0.0, "fit": 0.0, "predict": 0.0, "evaluate": 0.0}

        log.info(f"Initialized CNN model with patch_size={patch_size}, batch_size={batch_size}, epochs={epochs}")

    def _extract_patches(self, rgb_image, nir_image):
        """
        Extract 50x50 patches from RGB and NIR images.

        Args:
            rgb_image: numpy array of shape (H, W, 3)
            nir_image: numpy array of shape (H, W)

        Returns:
            patches_rgb: list of numpy arrays (50, 50, 3)
            patches_nir: list of numpy arrays (50, 50)
        """
        h, w = rgb_image.shape[:2]
        patches_rgb = []
        patches_nir = []

        for i in range(0, h - self.patch_size + 1, self.patch_size):
            for j in range(0, w - self.patch_size + 1, self.patch_size):
                patch_rgb = rgb_image[i : i + self.patch_size, j : j + self.patch_size]
                patch_nir = nir_image[i : i + self.patch_size, j : j + self.patch_size]

                if patch_rgb.shape[0] == self.patch_size and patch_rgb.shape[1] == self.patch_size:
                    patches_rgb.append(patch_rgb)
                    patches_nir.append(patch_nir)

        return patches_rgb, patches_nir

    def load_rgbi_image(self, file_path):
        """
        Load RGBI image and return RGB and NIR separately.

        Args:
            file_path: Path to RGBI TIFF file

        Returns:
            rgb_image: numpy array (H, W, 3)
            nir_image: numpy array (H, W)
        """
        import rasterio

        with rasterio.open(file_path) as src:
            data = src.read()  # Shape: (4, H, W)

            rgb = data[:3].transpose(1, 2, 0)  # (H, W, 3)
            nir = data[3]  # (H, W)

        return rgb.astype(np.uint8), nir.astype(np.float32)

    def train_from_data_loader(self, data_loader, max_files=None):
        """
        Load training data from DataLoader and extract patches.

        Args:
            data_loader: DataLoader instance
            max_files: Maximum number of files to process
        """
        log.info(f"Loading training data from {len(data_loader)} files")
        log.debug("Initial memory state:")
        log_cuda_info()

        start_time = time.time()

        files_processed = 0
        total_patches = 0

        for file_path, _is_temp in data_loader.iterate_files():
            if max_files and files_processed >= max_files:
                log.debug(f"Reached max_files limit ({max_files}), stopping")
                break

            try:
                log.info(f"Processing file {files_processed + 1}: {file_path}")
                log.debug(f"Current patches in memory: RGB={len(self.patches_rgb)}, NIR={len(self.patches_nir)}")
                log.debug(f"Estimated memory usage: {len(self.patches_rgb) * 50 * 50 * 3 / 1024**2:.2f} MB")

                rgb_image, nir_image = self.load_rgbi_image(file_path)
                log.debug(f"Loaded image shapes - RGB: {rgb_image.shape}, NIR: {nir_image.shape}")

                patches_rgb, patches_nir = self._extract_patches(rgb_image, nir_image)
                log.debug(f"Extracted {len(patches_rgb)} patches from current image")

                self.patches_rgb.extend(patches_rgb)
                self.patches_nir.extend(patches_nir)

                self.training_files.add(str(file_path))
                files_processed += 1
                total_patches += len(patches_rgb)

                log.info(f"  Extracted {len(patches_rgb)} patches from {file_path}")
                log.debug(f"  Total patches so far: {total_patches}")

                if files_processed % 5 == 0:
                    log_cuda_info()

            except Exception as e:
                log.error(f"Error processing {file_path}: {e}")
                continue

        duration = time.time() - start_time
        self.timing["processing"] = duration

        log.info(f"Loaded {total_patches} patches from {files_processed} files in {duration:.2f}s")
        log.debug(f"Final patches in memory: RGB={len(self.patches_rgb)}, NIR={len(self.patches_nir)}")
        log.debug(f"Final memory estimate: {len(self.patches_rgb) * 50 * 50 * 3 / 1024**2:.2f} MB")
        log_cuda_info()

    def fit_model(self):
        """Train the CNN model on collected patches."""
        log.info(f"Training CNN model on {len(self.patches_rgb)} patches")
        log_cuda_info()

        if len(self.patches_rgb) == 0:
            log.error("No training data available")
            return

        start_time = time.time()

        log.debug(f"Converting {len(self.patches_rgb)} patches to numpy arrays...")
        log.debug(f"Estimated memory for RGB patches: {len(self.patches_rgb) * 50 * 50 * 3 / 1024**3:.2f} GB")
        log.debug(f"Estimated memory for NIR patches: {len(self.patches_nir) * 50 * 50 / 1024**3:.2f} GB")

        patches_rgb = np.array(self.patches_rgb)
        log.debug(
            f"RGB patches array shape: {patches_rgb.shape}, dtype: {patches_rgb.dtype}, size: {patches_rgb.nbytes / 1024**3:.2f} GB"
        )
        log_cuda_info()

        patches_nir = np.array(self.patches_nir)
        log.debug(
            f"NIR patches array shape: {patches_nir.shape}, dtype: {patches_nir.dtype}, size: {patches_nir.nbytes / 1024**3:.2f} GB"
        )
        log_cuda_info()

        log.debug("Splitting into training and validation sets...")
        n_samples = len(patches_rgb)
        n_val = int(n_samples * self.validation_split)
        n_train = n_samples - n_val
        log.debug(f"n_train: {n_train}, n_val: {n_val}")
        log.debug(f"n_train: {n_train}, n_val: {n_val}")

        indices = np.random.permutation(n_samples)
        log.debug("Generated random permutation of indices")

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        log.debug(f"Split indices into train ({len(train_indices)}) and val ({len(val_indices)})")

        log.debug("Creating training dataset...")
        train_dataset = PatchDataset(patches_rgb[train_indices], patches_nir[train_indices])
        log.debug(f"Training dataset created with {len(train_dataset)} samples")
        log_cuda_info()

        log.debug("Creating validation dataset...")
        val_dataset = PatchDataset(patches_rgb[val_indices], patches_nir[val_indices])
        log.debug(f"Validation dataset created with {len(val_dataset)} samples")
        log_cuda_info()

        log.debug(f"Creating training DataLoader (batch_size={self.batch_size}, workers={self.num_workers})...")
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        log.debug(f"Training DataLoader created with {len(train_loader)} batches")
        log_cuda_info()

        log.debug(f"Creating validation DataLoader (batch_size={self.batch_size}, workers={self.num_workers})...")
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        log.debug(f"Validation DataLoader created with {len(val_loader)} batches")
        log_cuda_info()
        log_cuda_info()

        log.info(f"Training samples: {n_train}, Validation samples: {n_val}")

        log.debug("Starting training loop...")
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(self.epochs):
            log.debug(f"===== Epoch {epoch + 1}/{self.epochs} =====")
            log_cuda_info()

            log.debug("Starting training phase...")
            self.model.train()
            train_loss = 0.0

            for batch_idx, (rgb, nir) in enumerate(train_loader):
                if batch_idx == 0:
                    log.debug(f"First batch - RGB shape: {rgb.shape}, NIR shape: {nir.shape}")
                    log.debug(f"First batch - RGB dtype: {rgb.dtype}, NIR dtype: {nir.dtype}")
                    log.debug(f"Moving first batch to device: {self.device}")

                rgb = rgb.to(self.device)
                nir = nir.to(self.device)

                if batch_idx == 0:
                    log.debug("First batch moved to device")
                    log_cuda_info()

                self.optimizer.zero_grad()

                if batch_idx == 0:
                    log.debug("Running forward pass on first batch...")

                output = self.model(rgb)

                if batch_idx == 0:
                    log.debug(f"Forward pass complete, output shape: {output.shape}")
                    log_cuda_info()

                loss = self.criterion(output, nir)

                if batch_idx == 0:
                    log.debug(f"Loss computed: {loss.item():.6f}")

                loss.backward()

                if batch_idx == 0:
                    log.debug("Backward pass complete")
                    log_cuda_info()

                self.optimizer.step()

                if batch_idx == 0:
                    log.debug("Optimizer step complete")
                    log_cuda_info()

                train_loss += loss.item()

                if batch_idx % 100 == 0:
                    log.info(
                        f"Epoch [{epoch + 1}/{self.epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}"
                    )
                    if batch_idx % 500 == 0:
                        log_cuda_info()
                    if batch_idx % 500 == 0:
                        log_cuda_info()

            train_loss /= len(train_loader)
            log.debug(f"Training phase complete. Average train loss: {train_loss:.6f}")
            log_cuda_info()

            log.debug("Starting validation phase...")
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_batch_idx, (rgb, nir) in enumerate(val_loader):
                    if val_batch_idx == 0:
                        log.debug(f"Validation - first batch shape: RGB {rgb.shape}, NIR {nir.shape}")

                    rgb = rgb.to(self.device)
                    nir = nir.to(self.device)

                    output = self.model(rgb)
                    loss = self.criterion(output, nir)
                    val_loss += loss.item()

                    if val_batch_idx == 0:
                        log.debug(f"Validation - first batch loss: {loss.item():.6f}")
                        log_cuda_info()

            val_loss /= len(val_loader)
            log.debug(f"Validation phase complete. Average val loss: {val_loss:.6f}")
            log_cuda_info()
            log_cuda_info()

            log.info(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Stop early if validation loss does not improve.
            if val_loss < best_val_loss - 0.0001:
                log.debug(f"New best validation loss: {val_loss:.6f} (previous: {best_val_loss:.6f})")
                best_val_loss = val_loss
                patience_counter = 0
                log.debug("Saving best model checkpoint...")
                torch.save(self.model.state_dict(), "best_cnn_model.pth")
                log.info(f"New best model saved with validation loss: {val_loss:.6f}")
                log_cuda_info()
            else:
                patience_counter += 1
                log.debug(f"No improvement. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    log.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        log.debug("Loading best model from checkpoint...")
        self.model.load_state_dict(torch.load("best_cnn_model.pth"))
        log.debug("Best model loaded")
        log_cuda_info()

        duration = time.time() - start_time
        self.timing["fit"] = duration
        log.info(f"Model training complete in {duration:.2f}s")

    def predict_image(self, rgb_image):
        """
        Predict NIR for a full RGB image using patch-based prediction.

        Args:
            rgb_image: numpy array of shape (H, W, 3)

        Returns:
            nir_prediction: numpy array of shape (H, W)
        """
        log.info(f"Predicting NIR for image of shape {rgb_image.shape}")
        start_time = time.time()

        self.model.eval()

        h, w = rgb_image.shape[:2]
        nir_prediction = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, h - self.patch_size + 1, self.patch_size):
                for j in range(0, w - self.patch_size + 1, self.patch_size):
                    patch_rgb = rgb_image[i : i + self.patch_size, j : j + self.patch_size]

                    if patch_rgb.shape[0] == self.patch_size and patch_rgb.shape[1] == self.patch_size:
                        patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                        patch_tensor = patch_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

                        output = self.model(patch_tensor)
                        output_np = output.squeeze().cpu().numpy() * 255.0  # Denormalize

                        nir_prediction[i : i + self.patch_size, j : j + self.patch_size] += output_np
                        count_map[i : i + self.patch_size, j : j + self.patch_size] += 1

                        del patch_tensor, output

            for i in range(0, h, self.patch_size // 2):
                for j in range(0, w, self.patch_size // 2):
                    if i + self.patch_size > h or j + self.patch_size > w:
                        i_end = min(i + self.patch_size, h)
                        j_end = min(j + self.patch_size, w)
                        i_start = i_end - self.patch_size
                        j_start = j_end - self.patch_size

                        if i_start >= 0 and j_start >= 0:
                            patch_rgb = rgb_image[i_start:i_end, j_start:j_end]

                            if patch_rgb.shape[0] == self.patch_size and patch_rgb.shape[1] == self.patch_size:
                                patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                                patch_tensor = patch_tensor.unsqueeze(0).to(self.device)

                                output = self.model(patch_tensor)
                                output_np = output.squeeze().cpu().numpy() * 255.0  # Denormalize

                                nir_prediction[i_start:i_end, j_start:j_end] += output_np
                                count_map[i_start:i_end, j_start:j_end] += 1

                                # Clean up tensors
                                del patch_tensor, output

        # Average overlapping patch predictions to smooth seams.
        nir_prediction = np.divide(nir_prediction, count_map, where=count_map > 0)

        torch.cuda.empty_cache()

        duration = time.time() - start_time
        self.timing["predict"] += duration
        log.info(f"Prediction took {duration:.2f}s")

        return nir_prediction

    def load_model(self, model_path):
        """
        Load a pre-trained model from disk.

        Args:
            model_path: Path to the saved model file (.pth)
        """
        log.info(f"Loading model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        log.info(f"Model loaded from {model_path}")

    def evaluate_files(self, data_loader, max_files=None):
        """
        Evaluate the model on test files.

        Args:
            data_loader: DataLoader instance
            max_files: Maximum number of files to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        import gc

        from skimage.metrics import structural_similarity as ssim

        log.info("Evaluating model on test files")
        start_time = time.time()

        # Accumulate running stats to avoid storing every pixel.
        sum_squared_error = 0.0
        sum_abs_error = 0.0
        sum_true = 0.0
        sum_pred = 0.0
        sum_true_squared = 0.0
        sum_pred_squared = 0.0
        sum_true_pred = 0.0
        total_pixels = 0

        ssim_scores = []

        files_evaluated = 0

        for file_path, _is_temp in data_loader.iterate_files():
            # Skip files that were already seen during training.
            if str(file_path) in self.training_files:
                log.info(f"Skipping training file: {file_path}")
                continue

            if max_files and files_evaluated >= max_files:
                break

            try:
                log.info(f"Evaluating file {files_evaluated + 1}: {file_path}")
                rgb_image, nir_true = self.load_rgbi_image(file_path)

                nir_pred = self.predict_image(rgb_image)

                nir_true_flat = nir_true.flatten()
                nir_pred_flat = nir_pred.flatten()

                # Update metrics incrementally to keep memory usage low.
                sum_squared_error += np.sum((nir_true_flat - nir_pred_flat) ** 2)
                sum_abs_error += np.sum(np.abs(nir_true_flat - nir_pred_flat))
                sum_true += np.sum(nir_true_flat)
                sum_pred += np.sum(nir_pred_flat)
                sum_true_squared += np.sum(nir_true_flat**2)
                sum_pred_squared += np.sum(nir_pred_flat**2)
                sum_true_pred += np.sum(nir_true_flat * nir_pred_flat)
                total_pixels += len(nir_true_flat)

                data_range = nir_true.max() - nir_true.min()
                if data_range > 0:
                    ssim_score = ssim(nir_true, nir_pred, data_range=data_range)
                    ssim_scores.append(ssim_score)
                    log.info(f"  SSIM: {ssim_score:.4f}")

                files_evaluated += 1

                del rgb_image, nir_true, nir_pred, nir_true_flat, nir_pred_flat
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                log.error(f"Error evaluating {file_path}: {e}")
                continue

        if total_pixels == 0:
            log.error("No predictions made")
            return None

        mse = sum_squared_error / total_pixels
        rmse = np.sqrt(mse)

        mae = sum_abs_error / total_pixels

        mean_true = sum_true / total_pixels
        ss_tot = sum_true_squared - 2 * mean_true * sum_true + total_pixels * mean_true**2

        r2 = 1 - sum_squared_error / ss_tot if ss_tot > 0 else 0.0

        mean_ssim = np.mean(ssim_scores) if ssim_scores else 0.0

        duration = time.time() - start_time
        self.timing["evaluate"] = duration

        results = {
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae,
            "SSIM": mean_ssim,
            "n_files": files_evaluated,
        }

        log.info(f"Evaluation complete in {duration:.2f}s")

        return results


# Main Function.


@profile_execution
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate CNN NIR prediction model")
    parser.add_argument(
        "-n",
        type=int,
        help="Number of files to use for training/evaluation",
        default=None,
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Path to RGB image file for prediction",
        default=None,
    )
    parser.add_argument("--output", type=str, help="Path to save predicted NIR image", default=None)
    parser.add_argument(
        "--data-source",
        type=str,
        help="Path to file list (.txt) or directory containing ZIP files",
        default="/home/klugej/thesis/rgbi_files.txt",
    )
    parser.add_argument(
        "--eval-data-source",
        type=str,
        help="Separate data source for evaluation (optional)",
        default=None,
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        help="Fraction of files to use for training (rest for evaluation)",
        default=0.8,
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for training", default=32)
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=20)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", default=0.001)
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loading workers (default: auto-detect from CPUs)",
        default=None,
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to pre-trained model file (.pth) to load and evaluate",
        default=None,
    )
    args = parser.parse_args()

    model = CNNNir(
        patch_size=50,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=0.2,
        num_workers=args.num_workers,
    )

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from DataLoader import DataLoader

    if args.load_model:
        log.info(f"Load model mode: Loading pre-trained model from {args.load_model}")
        try:
            model.load_model(args.load_model)

            eval_source = args.eval_data_source or args.data_source
            log.info(f"Evaluating model on data from: {eval_source}")

            with DataLoader(eval_source) as eval_data_loader:
                log.info(f"Evaluation DataLoader initialized: {eval_data_loader}")
                eval_results = model.evaluate_files(eval_data_loader, max_files=args.n)

                if eval_results is not None:
                    print("\n" + "=" * 50)
                    print("EVALUATION RESULTS")
                    print("=" * 50)
                    print(f"  RMSE:  {eval_results['RMSE']:.4f}")
                    print(f"  R²:    {eval_results['R2']:.4f}")
                    print(f"  MAE:   {eval_results['MAE']:.4f}")
                    print(f"  SSIM:  {eval_results['SSIM']:.4f}")
                    print(f"  Files: {eval_results['n_files']}")
                    print("=" * 50)
                else:
                    log.error("Evaluation failed to produce results")

        except Exception as e:
            log.error(f"Failed to load model or evaluate: {e}")
            import traceback

            traceback.print_exc()

        return

    try:
        if args.eval_data_source:
            log.info(f"Using separate data sources - Training: {args.data_source}, Evaluation: {args.eval_data_source}")

            with DataLoader(args.data_source) as data_loader:
                log.info(f"Training DataLoader initialized: {data_loader}")
                model.train_from_data_loader(data_loader, max_files=args.n)
                model.fit_model()
                print(f"Training complete. Patches used: {len(model.patches_rgb)}")

            # Evaluate on separate evaluation data source
            try:
                with DataLoader(args.eval_data_source) as eval_data_loader:
                    log.info(f"Evaluation DataLoader initialized: {eval_data_loader}")
                    eval_results = model.evaluate_files(eval_data_loader, max_files=min(args.n or 20, 20))
                    if eval_results is not None:
                        print("Evaluation results:")
                        print(f"  RMSE: {eval_results['RMSE']:.4f}")
                        print(f"  R²:   {eval_results['R2']:.4f}")
                        print(f"  MAE:  {eval_results['MAE']:.4f}")
                        print(f"  SSIM: {eval_results['SSIM']:.4f}")
            except Exception as e:
                log.error(f"Evaluation failed: {e}")

        else:
            log.info(f"Using single data source with automatic train/test split: {args.data_source}")

            with DataLoader(args.data_source) as data_loader:
                log.info(f"DataLoader initialized: {data_loader}")

                total_files = len(data_loader)
                max_train_files = int(total_files * args.train_test_split) if args.n is None else args.n

                log.info(
                    f"Total files: {total_files}, Using {max_train_files} for training ({args.train_test_split:.1%} split)"
                )

                model.train_from_data_loader(data_loader, max_files=max_train_files)
                model.fit_model()
                print(f"Training complete. Patches used: {len(model.patches_rgb)}")

            # Evaluate on remaining files
            try:
                with DataLoader(args.data_source) as eval_data_loader:
                    log.info(f"Created separate DataLoader for evaluation: {eval_data_loader}")
                    log.info(f"Evaluation will automatically skip {len(model.training_files)} files used in training")
                    eval_results = model.evaluate_files(eval_data_loader, max_files=min(args.n or 20, 20))
                    if eval_results is not None:
                        print("Evaluation results:")
                        print(f"  RMSE: {eval_results['RMSE']:.4f}")
                        print(f"  R²:   {eval_results['R2']:.4f}")
                        print(f"  MAE:  {eval_results['MAE']:.4f}")
                        print(f"  SSIM: {eval_results['SSIM']:.4f}")
                        print(f"  Files used for evaluation: {eval_results['n_files']}")
            except Exception as e:
                log.error(f"Evaluation failed: {e}")

    except Exception as e:
        log.error(f"Failed to initialize DataLoader: {e}")
        return

    if args.predict and args.output:
        log.info(f"Predicting NIR for {args.predict} and saving to {args.output}")
        try:
            rgb_image, _ = model.load_rgbi_image(args.predict)
            nir_pred = model.predict_image(rgb_image)

            # Save prediction
            import rasterio

            with rasterio.open(args.predict) as src:
                profile = src.profile
                profile.update(count=1, dtype=rasterio.float32)

                with rasterio.open(args.output, "w", **profile) as dst:
                    dst.write(nir_pred.astype(np.float32), 1)

            log.info("NIR prediction and saving completed")
        except Exception as e:
            log.error(f"NIR prediction or saving failed: {e}")
    elif args.predict or args.output:
        log.error("Both --predict and --output arguments must be provided to generate NIR prediction")


if __name__ == "__main__":
    main()
