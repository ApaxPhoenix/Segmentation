import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
import warnings
import logging
import numpy as np

# Use the logger that was initialized in main
logger = logging.getLogger("loader")


def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """Combines individual data samples into a single batch for training.

    Images sometimes fail to load and return None. This function removes
    failed samples and stacks successful ones into proper batch tensors
    that PyTorch can work with.

    Args:
        batch: Collection of (image, mask) pairs. Failed samples will be None.

    Returns:
        Batch tensors ready for model training, or None if all samples failed.
    """
    # Remove any samples that failed during loading
    batch: List[Tuple[torch.Tensor, torch.Tensor]] = [item for item in batch if item is not None]

    # Check if we lost everything
    if not batch:
        warnings.warn(message="Complete batch failure - no samples loaded successfully")
        logger.warning("All samples in batch returned None")
        return None

    # Split images and masks for separate stacking
    images: Tuple[torch.Tensor, ...]
    masks: Tuple[torch.Tensor, ...]
    images, masks = zip(*batch)

    # Combine into batch tensors
    return torch.stack(tensors=images), torch.stack(tensors=masks)


class DatasetLoader(Dataset):
    """Loads images with XML polygon annotations for segmentation tasks.

    Designed for datasets where object boundaries are defined by polygon
    coordinates in XML files. Converts these polygons into pixel-level
    segmentation masks for training neural networks.

    Expected directory structure:
        dataset_root/
        ├── images/          # Image files (.jpg, .png, etc.)
        └── annotations/     # XML files matching image names

    Attributes:
        images: Path to image directory
        annotations: Path to XML annotation directory
        transform: Optional preprocessing pipeline
        files: Matched (image, xml) pairs
        classes: Mapping from class names to integer IDs
    """

    def __init__(
            self, dirpath: Path, transform: Optional[transforms.Compose] = None
    ) -> None:
        """Initialize dataset by discovering matching image-annotation pairs.

        Scans the dataset directory to find images with corresponding XML
        files, then analyzes all annotations to build a class vocabulary.

        Args:
            dirpath: Root dataset directory containing 'images' and 'annotations'
            transform: Optional preprocessing transforms (resize, normalize, etc.)

        Raises:
            UserWarning: When required directories are missing or empty
        """
        # Set up directory paths
        self.images: Path = Path(dirpath, "images")
        self.annotations: Path = Path(dirpath, "annotations")
        self.transform: Optional[transforms.Compose] = transform

        # Verify images directory exists
        if not self.images.exists():
            warnings.warn(message=f"Images directory not found: {self.images}")
            logger.warning(f"Missing images directory: {self.images}")
            self.files: List[Tuple[Path, Path]] = []
            self.classes: Dict[str, int] = {}
            return

        # Verify annotations directory exists
        if not self.annotations.exists():
            warnings.warn(
                message=f"Annotations directory not found: {self.annotations}"
            )
            logger.warning(f"Missing annotations directory: {self.annotations}")
            self.files: List[Tuple[Path, Path]] = []
            self.classes: Dict[str, int] = {}
            return

        # Match image files with their XML annotations
        self.files: List[Tuple[Path, Path]] = [
            (image, annotation)
            for annotation in self.annotations.glob(pattern="*.xml")
            for pattern in ["*.jpg", "*.jpeg", "*.png", "*.tiff"]
            for image in self.images.glob(pattern=pattern)
            if image.stem == annotation.stem  # Filenames must match
        ]

        # Build class vocabulary by scanning all XML files
        self.classes: Dict[str, int] = {
            name: label
            for label, name in enumerate(
                dict.fromkeys(
                    cast(ET.Element, object.find(path="name")).text
                    for _, annotation in self.files
                    for object in ET.parse(source=annotation)
                    .getroot()
                    .findall(path="object")
                )
            )
        }

        # Warn if no valid pairs were found
        if not self.files:
            warnings.warn(
                message="No matching image-annotation pairs discovered. Check filenames!"
            )
            logger.warning("Zero valid pairs found - verify naming convention")
            return

        logger.info(f"Loaded {len(self.files)} image-annotation pairs")
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            Count of valid image-annotation pairs
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor], None]:
        """Load and process a single sample from the dataset.

        Args:
            index: Sample index to retrieve (0 to len-1)

        Returns:
            Tuple of (image, mask) or None if loading fails
        """
        # Validate index bounds
        if index >= len(self.files):
            logger.error(f"Index {index} exceeds dataset size ({len(self.files)})")
            return None

        # Get file paths for this sample
        image: Path
        annotation: Path
        image, annotation = self.files[index]

        try:
            # Load image and get original dimensions
            image: Image.Image = Image.open(fp=image)
            width: int
            height: int
            width, height = image.size

            # Parse XML annotation file
            tree: ET.ElementTree = ET.parse(source=annotation)
            root: ET.Element = tree.getroot()

            # Create blank segmentation mask
            mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            canvas: Image.Image = Image.fromarray(mask)
            draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)

            # Find all annotated objects
            objects: List[ET.Element] = root.findall(path="object")

            # Extract class labels for each object
            labels: List[int] = [
                self.classes[cast(ET.Element, object.find(path="name")).text]
                for object in objects
            ]

            # Extract polygon coordinates for each object
            polygons: List[List[Tuple[float, float]]] = [
                [(float(cast(ET.Element, polygon)[i].text),
                  float(cast(ET.Element, polygon)[i + 1].text))
                 for i in range(0, len(cast(ET.Element, polygon)), 2)]
                for polygon in [cast(ET.Element, object.find("polygon")) for object in objects]
            ]

            # Render each polygon onto the mask using its class ID
            for label, points in zip(labels, polygons):
                draw.polygon(points, fill=label)

            # Convert mask back to numpy array
            mask = np.array(canvas)

            # Apply preprocessing transforms if provided
            if self.transform:
                image: Tensor = self.transform(image)

                # Determine new dimensions after transformation
                mheight: int = height
                mwidth: int = width
                for transform in self.transform.transforms:
                    if hasattr(transform, 'size'):
                        mheight, mwidth = transform.size
                        break

                # Calculate resize scaling factors
                scaley: np.ndarray = np.divide(mheight, height)
                scalex: np.ndarray = np.divide(mwidth, width)

                # Generate coordinate arrays for new dimensions
                yindex: np.ndarray = np.arange(mheight)
                xindex: np.ndarray = np.arange(mwidth)

                # Map new coordinates back to original image space
                origy: np.ndarray = np.divide(yindex, scaley).astype(np.int32)
                origx: np.ndarray = np.divide(xindex, scalex).astype(np.int32)

                # Clamp coordinates to stay within original bounds
                origy = np.clip(origy, 0, np.subtract(height, 1))
                origx = np.clip(origx, 0, np.subtract(width, 1))

                # Resize mask using coordinate remapping
                ygrid: np.ndarray
                xgrid: np.ndarray
                ygrid, xgrid = np.meshgrid(origy, origx, indexing='ij')
                mask = mask[ygrid, xgrid]
            else:
                # No transforms - just convert to tensor
                image: Tensor = transforms.ToTensor()(image)

            # Convert mask to PyTorch tensor (long dtype for class indices)
            mask: torch.Tensor = torch.from_numpy(mask).long()

            return image, mask

        except Exception as error:
            # Log the error and return None to skip this sample
            logger.error(f"Sample {index} loading failed ({image}): {error}")
            warnings.warn(f"Skipping sample {index}: {str(error)}")
            return None