import os
import random
from pathlib import Path

# City1 -> Img1-Mask1, Img2-Mask2, ...
class ImgMaskLoader:
    """
    A class to pair images with corresponding masks for a specific city.
    """
    IMG_TYPE = "leftImg8bit"
    IMG_FORMAT = ".png"
    MASK_TYPE = "labelIds"
    MASK_FORMAT = ".png"

    def __init__(self, img_path, mask_path):
        """
        Initialize the ImgMaskLoader with paths to images and masks.

        Args:
            img_path (str): Path to the directory containing image files.
            mask_path (str): Path to the directory containing mask files.
        """
        self.img_path = img_path
        self.mask_path = mask_path

        self.img_list = sorted([filename for filename in os.listdir(self.img_path)])
        self.mask_list = sorted([
            filename
            for filename in os.listdir(self.mask_path)
                if filename.split('_')[-1] == self.MASK_TYPE + self.MASK_FORMAT
        ])

        self._run_tests()

    def _run_tests(self):
        """
        Validate that the images and masks are consistent in number and naming.
        """
        if len(self.img_list) != len(self.mask_list):
            raise ValueError(
                f"Number of images ({len(self.img_list)}) and masks ({len(self.mask_list)}) do not match."
            )

        for img_file, mask_file in zip(self.img_list, self.mask_list):
            img_parts = img_file.split('_')
            mask_parts = mask_file.split('_')
            if img_parts[:3] != mask_parts[:3]:  # Match first 3 parts of the name
                raise ValueError(
                    f"Image and mask do not align: {img_file.name} vs {mask_file.name}."
                )

    def __iter__(self):
        """
        Return an iterator over image-mask pairs.

        Returns:
            Iterator[Tuple[Path, Path]]: An iterator of (image_path, mask_path) pairs.
        """
        self._index = 0
        return self

    def __next__(self):
        """
        Retrieve the next image-mask pair.

        Returns:
            Tuple[Path, Path]: A tuple containing the image path and mask path.

        Raises:
            StopIteration: When all image-mask pairs are exhausted.
        """
        while self._index < len(self.img_list):
            result = (
                Path(self.img_path, self.img_list[self._index]),
                Path(self.mask_path, self.mask_list[self._index])
            )
            self._index += 1
            return result
        raise StopIteration


class CityLoader:
    def __init__(self, subset: str, image_path: Path, mask_path: Path):
        """
        Initialize the CityLoader class for a given subset (e.g., 'train', 'valid', 'test').

        Args:
            subset (str): The subset of the data ('train', 'valid', or 'test').
            image_path (str): Base path to the image directory.
            mask_path (str): Base path to the mask directory.
        """
        self.subset = subset
        self.image_path = Path(image_path, subset)
        self.mask_path = Path(mask_path, subset)

        # Ensure the directories exist and have matching contents
        self._validate_city_directories()

        self.city_list = os.listdir(self.image_path)


    def _validate_city_directories(self):
        """
        Validates that the image and mask directories exist, have matching names, and have the same number of entries.
        """
        if not self.image_path.exists() or not self.image_path.is_dir():
            raise FileNotFoundError(f"Image path '{self.image_path}' does not exist or is not a directory.")
        if not self.mask_path.exists() or not self.mask_path.is_dir():
            raise FileNotFoundError(f"Mask path '{self.mask_path}' does not exist or is not a directory.")

        image_cities = os.listdir(self.image_path)
        mask_cities = os.listdir(self.mask_path)

        if len(image_cities) != len(mask_cities):
            raise ValueError(f"Mismatch in number of cities for subset '{self.subset}': "
                             f"{len(image_cities)} images vs {len(mask_cities)} masks.")
        if sorted(image_cities) != sorted(mask_cities):
            raise ValueError(f"Mismatch in city names for subset '{self.subset}'. "
                             f"Image cities: {image_cities}, Mask cities: {mask_cities}.")


    def get_img_mask_pairs(self, randomize=False) -> list[tuple[Path,Path]]:
        """
        Returns a list of tuples containing paths to the images and masks for all cities.

        Args:
            randomize (bool): Whether to shuffle the list of image-mask pairs.

        Returns:
            List[Tuple[Path, Path]]: List of image-mask path pairs.
        """
        img_mask_pairs = [
            img_mask_pair
            for city_name in self.city_list
            for img_mask_pair in ImgMaskLoader(Path(self.image_path, city_name), Path(self.mask_path, city_name))
        ]

        if randomize:
            random.shuffle(img_mask_pairs)

        return img_mask_pairs

    def __getitem__(self, city_name: str) -> str:
        """
        Retrieves the image-mask loader for a specific city.

        Args:
            city_name (str): Name of the city.

        Returns:
            ImgMaskLoader: An instance of ImgMaskLoader for the given city.

        Raises:
            ValueError: If the city is not found in the list of cities.
        """
        if city_name in self.city_list:
            return ImgMaskLoader(Path(self.image_path, city_name), Path(self.mask_path, city_name))
        else:
            raise ValueError(f"City '{city_name}' not found. Available cities: {', '.join(self.city_list)}.")
