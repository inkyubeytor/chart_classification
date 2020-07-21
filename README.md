# chart_classification

A chart classification pipeline.

To Test:
import_images with URL
train_and_save
load_and_predict

Potential refactors:
Allow custom data store location
use only filenames in log files instead of paths

# Structure
```
# pipeline/dataset.py
new_dataset(filenames: List[str], conversions: List[str]) -> str
delete_dataset(dataset: str) -> None
make_data(dataset: str, transforms: List[str], bundled: bool = True) -> bool
get_process(dataset: str) -> Dict[str, Any]

# pipeline/store.py
CLASSES: Dict[str, int]
DEFAULT_CLASS: str
init_data_store() -> None
import_images(images: List[str], labels: Optional[List[str]] = None, urls: bool = False) -> None:
convert_images(images: List[str], conversions: List[str]) -> None

# pipeline/conversions.py
CONVERSIONS: Dict[str, Callable[[str, Optional[str]], str]]

# pipeline/transforms.py
TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]]

# pipeline/retrieval.py

# modelling.py
train_and_save(classifier: ClassifierMixin, dataset: str, transforms: List[str], bundled: bool) -> None
load_and_predict(model_dataset: str, test_dataset: str) -> None
```
