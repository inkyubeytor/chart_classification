# chart_classification

A chart classification pipeline.

TODO:
* Given a list of image names from data store and list of conversions, apply
  conversions to images in data store (destructively)
* Given list of image names from data store and list of conversions, make a new 
  dataset with those converted images
  * Assert those images have no other conversions applied
  * Copy the images, then destructively convert them
  * Record the set of conversions in process.json

* Train/Classify
  * Load training data
  * Train and save model
  * Load model
  * Predict with model



# Structure
```
# pipeline/construction.py
make_data(dataset: str, transforms: List[str], bundled: bool = True) -> bool

# pipeline/conversions.py
CONVERSIONS: Dict[str, Callable[[str, Optional[str]], str]]

# pipeline/retrieval.py
import_images(images: List[str], labels: Optional[List[str]] = None, urls: bool = False) -> None:

# pipeline/storage.py
CLASSES: Dict[str, int]
DEFAULT_CLASS: str
init_data_store() -> None
new_dataset() -> str
delete_dataset(dataset: str) -> None

# pipeline/transforms.py
TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]]
```