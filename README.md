# chart_classification

A chart classification pipeline.

TODO:
* Given a list of image names from data store and list of conversions, apply
  conversions to images in data store (destructively)


# Structure
```
# pipeline/dataset.py
new_dataset(filenames: List[str], conversions: List[str]) -> str
delete_dataset(dataset: str) -> None
make_data(dataset: str, transforms: List[str], bundled: bool = True) -> bool
get_process(dataset: str) -> Dict[str, Any]

# pipeline/storage.py
CLASSES: Dict[str, int]
DEFAULT_CLASS: str
init_data_store() -> None
import_images(images: List[str], labels: Optional[List[str]] = None, urls: bool = False) -> None:


# pipeline/conversions.py
CONVERSIONS: Dict[str, Callable[[str, Optional[str]], str]]

# pipeline/transforms.py
TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]]

# pipeline/retrieval.py

```
