import os
import random
import asyncio
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from loader import DatasetLoader, collate_fn
from pathlib import Path
import logging
import warnings
from bar import Bar
from typing import Dict, List, Literal, Optional, Tuple

# Get logger instance from main application
logger = logging.getLogger("trainer")


class Trainer:
    """Professional deep learning trainer for semantic segmentation models.

    This class implements a comprehensive training pipeline for semantic segmentation
    tasks using PyTorch. It manages the complete training lifecycle including data
    loading, model initialization, training loops, validation, testing, and model
    persistence with industry-standard best practices.

    The trainer supports advanced features such as distributed training, automatic
    overfitting detection, learning rate scheduling, gradient clipping, and
    comprehensive logging for production-ready model development.

    Args:
        module (nn.Module): Neural network module implementing segmentation architecture.
        training_path (Path): Directory path containing training dataset with images and masks.
        validation_path (Path): Directory path containing validation dataset with images and masks.
        testing_path (Path): Directory path containing test dataset with images and masks.
        weights_path (Optional[Path], optional): Path to pre-trained model weights for
            transfer learning. Defaults to None.
        dimensions (Tuple[int, int]): Target image dimensions (height, width) for input
            preprocessing.
        epochs (int): Total number of training epochs to execute.
        batch (int): Batch size for mini-batch gradient descent.
        lr (Optional[float], optional): Learning rate for Adam optimizer. Defaults to None.
        decay (Optional[float], optional): L2 weight decay regularization coefficient.
            Defaults to None.
        gamma (Optional[float], optional): Learning rate decay multiplier for scheduler.
            Defaults to None.
        workers (Optional[int], optional): Number of parallel processes for data loading.
            Defaults to None.
        seed (Optional[int], optional): Random seed for reproducible training results.
            Defaults to None.
        parallelism (Optional[bool], optional): Enable distributed training across multiple
            GPUs. Defaults to False.

    Attributes:
        module (nn.Module): The neural network model being trained.
        device (torch.device): Computing device (CUDA GPU or CPU) for tensor operations.
        dimensions (Tuple[int, int]): Input image dimensions for consistent preprocessing.
        epochs (int): Total training epochs configured.
        workers (Optional[int]): Number of data loading worker processes.
        training_dataset (DataLoader): DataLoader for training data with augmentations.
        validation_dataset (DataLoader): DataLoader for validation data without augmentations.
        testing_dataset (DataLoader): DataLoader for test data without augmentations.
        criterion (nn.CrossEntropyLoss): Loss function for pixel-wise classification.
        optimizer (torch.optim.Adam): Adam optimizer for parameter updates.
        scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler for adaptive
            learning rate adjustment.
        cache (Dict[str, List[float]]): Dictionary storing training and validation loss history.

    Raises:
        TypeError: If module is not an instance of nn.Module.
        ValueError: If any path is invalid or hyperparameters are incorrectly specified.
    """

    def __init__(
            self,
            module: nn.Module,
            training_path: Path,
            validation_path: Path,
            testing_path: Path,
            weights_path: Optional[Path] = None,
            dimensions: Tuple[int, int] = (224, 224),
            epochs: int = 100,
            batch: int = 32,
            lr: Optional[float] = 1e-3,
            decay: Optional[float] = 1e-4,
            gamma: Optional[float] = 0.9,
            workers: Optional[int] = None,
            seed: Optional[int] = None,
            parallelism: Optional[bool] = False,
    ) -> None:
        """Initialize the segmentation trainer with comprehensive validation and setup.

        Performs extensive validation of input parameters, configures the training
        environment, sets up data loaders with appropriate transforms, initializes
        the model with proper weight initialization, and prepares optimization components.

        Raises:
            TypeError: If module is not a valid nn.Module instance.
            ValueError: If paths are invalid or hyperparameters are incorrectly specified.
        """
        # Validate neural network module type and structure
        if not isinstance(module, nn.Module):
            logger.error(f"Expected 'module' to be a nn.Module instance, but got {type(module)}.")
            raise TypeError(
                f"Expected 'module' to be a nn.Module instance, but got {type(module)}."
            )

        # Validate dataset directory paths for accessibility and existence
        if not isinstance(training_path, Path) or not training_path.exists():
            logger.error(f"Invalid training data path: {training_path}")
            raise ValueError(f"Invalid training data path: {training_path}")
        if not isinstance(validation_path, Path) or not validation_path.exists():
            logger.error(f"Invalid validation data path: {validation_path}")
            raise ValueError(f"Invalid validation data path: {validation_path}")
        if not isinstance(testing_path, Path) or not testing_path.exists():
            logger.error(f"Invalid test data path: {testing_path}")
            raise ValueError(f"Invalid test data path: {testing_path}")

        # Validate image dimensions tuple for preprocessing consistency
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            logger.error(f"Input dimensions should be a tuple of two integers (H, W), got: {dimensions}")
            raise ValueError(
                f"Input dimensions should be a tuple of two integers (H, W), got: {dimensions}"
            )

        # Validate training hyperparameter ranges and types
        if not isinstance(epochs, int) or epochs <= 0:
            logger.error(f"Training epochs should be a positive integer, got: {epochs}")
            raise ValueError(f"Training epochs should be a positive integer, got: {epochs}")
        if not isinstance(batch, int) or batch <= 0:
            logger.error(f"Batch size should be a positive integer, got: {batch}")
            raise ValueError(f"Batch size should be a positive integer, got: {batch}")

        # Validate pre-trained weights path if provided for model initialization
        if weights_path and (
                not isinstance(weights_path, Path) or not weights_path.exists()
        ):
            logger.warning(f"Invalid model weights path: {weights_path}")
            warnings.warn(f"Invalid model weights path: {weights_path}")

        # Validate distributed training configuration parameter
        if parallelism is not None and not isinstance(parallelism, bool):
            logger.warning(f"Parallelism flag should be a boolean value, got: {type(parallelism)}.")
            warnings.warn(
                f"Parallelism flag should be a boolean value, got: {type(parallelism)}."
            )

        # Configure reproducible training environment with deterministic behavior
        if seed is not None:
            self.seed(seed=seed)
            logger.info(f"Random seed set to: {seed}")

        # Determine optimal computing device based on CUDA availability
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Initialize neural network model and transfer to target device
        self.module = module.to(self.device)
        self.dimensions = dimensions
        self.epochs = epochs
        self.workers = workers

        # Initialize model weights using appropriate strategy
        if weights_path is None:
            self.initialize_weights(self.module)
            logger.info("Applied weight initialization to model")
        else:
            try:
                # Load pre-trained weights with error handling and state dict extraction
                state_dict = torch.load(weights_path, map_location=self.device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.module.load_state_dict(state_dict)
                logger.info(f"Loaded pre-trained weights from {weights_path}")
            except Exception as error:
                logger.warning(f"Error loading weights: {error}. Initializing randomly.")
                warnings.warn(f"Error loading weights: {error}. Initializing randomly.")
                self.initialize_weights(self.module)

        # Initialize training metrics cache for loss history tracking
        self.cache = {"training": [], "validation": []}

        # Configure distributed training across multiple GPUs if available and requested
        if parallelism and np.greater(torch.cuda.device_count(), 1):
            self.module = nn.parallel.DistributedDataParallel(self.module)
            logger.info(
                f"Using distributed training with {torch.cuda.device_count()} GPUs"
            )
        else:
            logger.info("Using single GPU/CPU training")

        # Initialize training data loader with augmentation transforms
        try:
            self.training_dataset = self.loader(
                dirpath=training_path, batch=batch, mode="training"
            )
            logger.info(f"Training data loaded successfully")
        except Exception as error:
            logger.warning(f"Error loading training data: {error}")
            warnings.warn(f"Error loading training data: {error}")

        # Initialize validation data loader with normalization-only transforms
        try:
            self.validation_dataset = self.loader(
                dirpath=validation_path, batch=batch, mode="validation"
            )
            logger.info(f"Validation data loaded successfully")
        except Exception as error:
            logger.warning(f"Error loading validation data: {error}")
            warnings.warn(f"Error loading validation data: {error}")

        # Initialize testing data loader for final model evaluation
        try:
            self.testing_dataset = self.loader(
                dirpath=testing_path, batch=batch, mode="testing"
            )
            logger.info(f"Test data loaded successfully")
        except Exception as error:
            logger.warning(f"Error loading test data: {error}")
            warnings.warn(f"Error loading test data: {error}")

        # Configure cross-entropy loss function for pixel-wise segmentation
        self.criterion = nn.CrossEntropyLoss()

        # Initialize Adam optimizer with specified learning rate and weight decay
        self.optimizer = torch.optim.Adam(
            params=self.module.parameters(),
            lr=lr,
            weight_decay=decay
        )

        # Configure step learning rate scheduler for adaptive learning rate decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=5,
            gamma=gamma,
        )

        logger.info(
            f"Module {module.__class__.__name__} initialized with lr={lr}, decay={decay}, gamma={gamma}"
        )
        logger.info(f"Training configuration: {epochs} epochs, batch size {batch}, image dims {dimensions}")

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """Apply layer-specific weight initialization for optimal training convergence.

        Implements proven initialization techniques tailored to different layer types
        commonly found in segmentation architectures. The initialization methods are
        selected based on theoretical foundations and empirical performance.

        Convolutional layers use Kaiming Normal initialization optimized for ReLU
        activations, batch normalization layers use unit weight and zero bias
        initialization, linear layers employ Xavier initialization, and recurrent
        layers use orthogonal initialization.

        Args:
            module (nn.Module): Neural network module to initialize with appropriate
                weight distributions.
        """
        for layer in module.modules():
            # Initialize 2D convolutional layers with Kaiming Normal for ReLU networks
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # Initialize transpose convolutions for decoder and upsampling paths
            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # Initialize batch normalization layers for stable gradient flow
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

            # Initialize fully connected layers with Xavier initialization
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            # Initialize LSTM layers with orthogonal weight matrices for stable gradients
            elif isinstance(layer, nn.LSTM):
                for parameter in layer.parameters():
                    if len(parameter.shape) >= 2:
                        nn.init.orthogonal_(parameter)
                    else:
                        nn.init.normal_(parameter)

            # Initialize GRU layers with orthogonal weight matrices for sequence stability
            elif isinstance(layer, nn.GRU):
                for parameter in layer.parameters():
                    if len(parameter.shape) >= 2:
                        nn.init.orthogonal_(parameter)
                    else:
                        nn.init.normal_(parameter)

    @staticmethod
    def seed(seed: int) -> None:
        """Configure deterministic training environment across all random generators.

        Establishes reproducible training conditions by setting seeds for Python's
        random module, NumPy, PyTorch CPU and CUDA operations, and CUDA backend
        configurations. This ensures consistent results across multiple training
        runs for scientific reproducibility and debugging purposes.

        Args:
            seed (int): Integer seed value for all random number generators.
        """
        try:
            # Configure Python hash randomization for consistent string hashing behavior
            os.environ["PYTHONHASHSEED"] = str(seed)

            # Set PyTorch random seed for reproducible tensor operations and initialization
            torch.manual_seed(seed=seed)

            # Configure Python random module for consistent data augmentation patterns
            random.seed(a=seed)

            # Set NumPy random seed for reproducible array operations and shuffling
            np.random.seed(seed=seed)

            # Configure CUDA-specific randomization settings if GPU is available
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed=seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        except Exception as error:
            logger.warning(f"Error setting seed: {str(error)}")
            warnings.warn(f"Error setting seed: {str(error)}")

    def loader(
            self,
            dirpath: Path,
            batch: int,
            mode: Literal["training", "validation", "testing"] = "training",
    ) -> Optional[DataLoader]:
        """Construct optimized DataLoader with mode-specific preprocessing transforms.

        Creates and configures a PyTorch DataLoader with appropriate data augmentation
        strategies based on the specified mode. Training mode includes augmentations
        for improved generalization, while validation and testing modes use only
        normalization transforms for consistent evaluation.

        Args:
            dirpath (Path): Path to dataset directory containing images and segmentation masks.
            batch (int): Number of samples per batch for mini-batch processing.
            mode (Literal["training", "validation", "testing"], optional): Operating mode
                determining appropriate preprocessing strategy. Defaults to "training".

        Returns:
            Optional[DataLoader]: Configured DataLoader instance or None if initialization fails.
        """
        # Configure mode-specific preprocessing and augmentation transforms
        if mode == "training":
            # Training transforms include augmentations for improved model generalization
            transform = transforms.Compose(
                [
                    transforms.Resize(size=self.dimensions),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            # Validation and testing use only normalization for consistent evaluation
            transform = transforms.Compose(
                [
                    transforms.Resize(size=self.dimensions),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        try:
            # Initialize dataset with custom loader and preprocessing transforms
            dataset = DatasetLoader(dirpath=dirpath, transform=transform)

            # Configure DataLoader with performance optimizations and parallelization
            return DataLoader(
                dataset=dataset,
                collate_fn=collate_fn,
                batch_size=batch,
                shuffle=(mode == "training"),
                num_workers=(4 if self.workers is None else self.workers),
                pin_memory=True,
            )
        except Exception as error:
            logger.warning(f"Error in data loading: {str(error)}")
            warnings.warn(f"Error in data loading: {str(error)}")
            return None

    async def rehearse(
            self, dataloader: DataLoader, mode: Literal["training", "validation"]
    ) -> float:
        """Execute single epoch of training or validation with comprehensive error handling.

        Processes all batches in the dataloader, computing forward passes, loss values,
        and optionally backward passes for parameter updates. Implements gradient
        clipping for training stability and maintains detailed progress tracking.

        Args:
            dataloader (DataLoader): DataLoader containing batched samples for processing.
            mode (Literal["training", "validation"]): Operating mode determining gradient
                computation and parameter updates.

        Returns:
            float: Average loss value across all batches in the epoch.
        """
        # Configure model mode based on training or validation phase
        self.module.train() if mode == "training" else self.module.eval()

        # Initialize loss accumulator with high precision floating point arithmetic
        total_loss = np.float64(0.0)

        # Process epoch with asynchronous progress visualization and timing
        async with Bar(iterations=len(dataloader), title=f"{mode.capitalize()}", steps=20) as bar:
            time = asyncio.get_event_loop().time()

            # Process each batch in the dataloader with comprehensive error handling
            for index, (inputs, targets) in enumerate(dataloader, start=1):
                # Validate tensor types to prevent runtime errors and data corruption
                if not isinstance(inputs, torch.Tensor):
                    logger.warning("Inputs must be torch tensors.")
                    warnings.warn("Inputs must be torch tensors.")
                    continue

                if not isinstance(targets, torch.Tensor):
                    logger.warning("Targets must be torch tensors.")
                    warnings.warn("Targets must be torch tensors.")
                    continue

                try:
                    # Transfer data tensors to target device for computation
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)

                    # Clear accumulated gradients from previous iteration
                    self.optimizer.zero_grad()

                    # Compute forward pass with conditional gradient computation based on mode
                    with torch.set_grad_enabled(mode=(mode == "training")):
                        outputs = self.module(inputs)
                        loss = self.criterion(outputs, targets)

                        # Validate loss tensor to prevent NaN propagation and training instability
                        if not isinstance(loss, torch.Tensor):
                            logger.warning("Loss must be a torch.Tensor.")
                            warnings.warn("Loss must be a torch.Tensor.")
                            continue

                        # Check for NaN loss indicating numerical instability or training issues
                        if torch.isnan(loss):
                            logger.warning("NaN loss detected! Skipping batch.")
                            warnings.warn("NaN loss detected! Skipping batch.")
                            continue

                        # Perform backward pass and parameter updates only in training mode
                        if mode == "training":
                            try:
                                loss.backward()
                                # Apply gradient clipping for training stability and convergence
                                torch.nn.utils.clip_grad_norm_(
                                    parameters=self.module.parameters(), max_norm=1.0
                                )
                                self.optimizer.step()
                            except Exception as error:
                                logger.warning(f"Error in backward pass: {str(error)}")
                                warnings.warn(f"Error in backward pass: {str(error)}")
                                continue

                    # Accumulate weighted loss for proper batch-size-adjusted averaging
                    total_loss = np.add(
                        total_loss,
                        np.multiply(np.float64(loss.item()), np.float64(len(inputs))),
                    )

                    # Update asynchronous progress visualization with current metrics
                    await bar.update(batch=index, time=time)
                    await bar.postfix(loss=np.divide(total_loss, index))

                except Exception as error:
                    logger.warning(f"Error processing batch: {str(error)}")
                    warnings.warn(f"Error processing batch: {str(error)}")
                    continue

            # Calculate epoch average loss across all processed batches
            average_loss = float(
                np.divide(total_loss, np.float64(len(dataloader)))
            )

            return average_loss

    async def train(self) -> None:
        """Execute complete training procedure with overfitting detection and scheduling.

        Orchestrates the full training process including epoch loops, validation cycles,
        performance monitoring, and model checkpointing. Implements overfitting detection
        by monitoring validation loss trends and automatically saves model checkpoints
        when overfitting is detected.

        The method includes comprehensive logging, learning rate scheduling, and error
        handling to maintain training stability across extended training sessions.
        """
        logger.info(f"Starting training for {self.epochs} epochs")

        # Execute training loop for specified number of epochs with monitoring
        for epoch in range(self.epochs):
            try:
                print(f"Epoch {epoch + 1}/{self.epochs}")
                logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")

                # Process training and validation phases sequentially for each epoch
                for mode, dataloader in [("training", self.training_dataset), ("validation", self.validation_dataset)]:
                    loss = await self.rehearse(dataloader=dataloader, mode=mode)

                    # Log epoch results for training monitoring and debugging
                    logger.info(f"Epoch {epoch + 1}/{self.epochs}, {mode.capitalize()} Loss: {loss:.4f}")

                    # Cache metrics for overfitting detection and performance analysis
                    self.cache[mode].append(loss)

                # Monitor for overfitting patterns after initial training epoch
                if np.greater(epoch, 0):
                    # Check for overfitting by comparing current and previous losses
                    # Overfitting occurs when validation loss increases while training loss decreases
                    if np.greater(
                            self.cache["validation"][-1],
                            self.cache["validation"][-2],
                    ) and np.less(
                        self.cache["training"][-1],
                        self.cache["training"][-2]
                    ):
                        logger.warning(
                            f"Overfitting detected at epoch {epoch + 1}: "
                            f"Validation loss increased while training loss decreased. "
                            f"Saving current model state."
                        )
                        warnings.warn(
                            f"Overfitting detected at epoch {epoch + 1}: "
                            f"Validation loss increased while training loss decreased. "
                            f"Saving current model state."
                        )
                        # Save checkpoint when overfitting is detected for model recovery
                        self.save(filepath=Path(f"checkpoints/epoch-{epoch + 1}.pth"))

                # Update learning rate schedule for adaptive learning rate adjustment
                self.scheduler.step()

                # Log current learning rate for training monitoring and debugging
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Current learning rate: {lr:.6f}")

            except Exception as error:
                logger.warning(f"Error in training epoch {epoch + 1}: {str(error)}")
                warnings.warn(f"Error in training epoch {epoch + 1}: {str(error)}")
                continue

        logger.info("Training completed successfully")

    async def test(self) -> None:
        """Conduct comprehensive testing evaluation with accuracy metrics computation.

        Performs final model evaluation on the test dataset, computing both loss and
        pixel-wise accuracy metrics. This method processes the entire test dataset
        and provides detailed performance statistics for model assessment.

        The testing process includes progress visualization, comprehensive error handling,
        and detailed logging of final performance metrics for model evaluation.
        """
        logger.info("Starting model evaluation on test dataset")

        # Configure model for evaluation mode without gradient computation
        self.module.eval()

        # Initialize metric accumulators for loss and accuracy computation
        total_loss = np.float64(0.0)
        all_predictions = np.array([], dtype=np.int64)
        all_targets = np.array([], dtype=np.int64)

        # Process test dataset with asynchronous progress visualization
        async with Bar(iterations=len(self.testing_dataset), title="Testing", steps=20) as bar:
            time = asyncio.get_event_loop().time()

            # Process each test batch with comprehensive error handling
            for batch, (inputs, targets) in enumerate(self.testing_dataset, start=1):
                try:
                    # Validate tensor types for test data integrity and consistency
                    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                        logger.warning(
                            f"Skipping batch: Expected torch.Tensor but got {type(inputs)}/{type(targets)}"
                        )
                        warnings.warn(
                            f"Skipping batch: Expected torch.Tensor but got {type(inputs)}/{type(targets)}"
                        )
                        continue

                    # Transfer data tensors to computation device for inference
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)

                    # Perform inference without gradient computation for efficiency
                    with torch.no_grad():
                        outputs = self.module(inputs)
                        loss = self.criterion(outputs, targets)

                        # Accumulate weighted loss for proper batch-size-adjusted averaging
                        total_loss = np.add(
                            total_loss,
                            np.multiply(np.float64(loss.item()), np.float64(inputs.size(0))),
                        )

                        # Extract predictions from model outputs for accuracy computation
                        _, prediction = torch.max(outputs, 1)

                        # Accumulate predictions and targets for final accuracy metrics
                        all_predictions = np.concatenate(
                            (all_predictions, prediction.cpu().numpy()), axis=0
                        )
                        all_targets = np.concatenate(
                            (all_targets, targets.cpu().numpy()), axis=0
                        )

                    # Update asynchronous progress visualization with current test metrics
                    await bar.update(batch, time)
                    await bar.postfix(loss=np.divide(total_loss, batch))

                except Exception as error:
                    logger.warning(f"Error processing test batch {batch}: {str(error)}")
                    warnings.warn(f"Error processing test batch {batch}: {str(error)}")
                    continue

        # Calculate final test metrics including loss and pixel-wise accuracy
        corrects = np.sum((all_predictions == all_targets))
        accuracy = np.multiply(
            np.divide(corrects, np.size(all_predictions)), 100
        )
        average_loss = float(np.divide(total_loss, len(self.testing_dataset)))

        # Display and log final test results for comprehensive model evaluation
        print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        logger.info(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def save(self, filepath: Optional[Path] = None) -> None:
        """Persist trained model to disk with flexible format support.

        Saves the trained segmentation model either as a complete model file or
        state dictionary based on the specified file extension. The method
        automatically creates necessary directories and provides appropriate
        feedback on the save operation.

        Args:
            filepath (Optional[Path], optional): Target path for model saving.
                Defaults to "model.pt" in current directory.
        """
        # Configure default save path if none provided by user
        if not filepath:
            parent = Path(__file__).parent
            filepath = Path(parent, "model.pt")
        else:
            # Ensure parent directories exist for custom save paths
            filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to: {filepath}")

        try:
            # Save model based on file extension preference
            if filepath.suffix == ".pth":
                # Save state dictionary for loading into same architecture later
                torch.save(obj=self.module.state_dict(), f=filepath)
                print(f"Model state dictionary saved at: {filepath}")
                logger.info(f"Model state dictionary successfully saved to: {filepath}")
            else:
                # Save complete model including architecture and parameters
                torch.save(obj=self.module, f=filepath)
                print(f"Model saved at: {filepath}")
                logger.info(f"Complete model successfully saved to: {filepath}")
                
        except Exception as error:
            logger.error(f"Error saving model: {str(error)}")
            warnings.warn(f"Error saving model: {str(error)}")
            raise