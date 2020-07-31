# chart_classification
A chart classification pipeline.

## Initialization
Initialization of the pipeline is done prior to any use. It should be done 
automatically, via `pipeline/__init__.py`. In cases where it is not, it can be
done manually by calling the `init_data_store` function of `pipeline/store.py`.

The initialization process creates a `data` subdirectory in the root directory
of the repository. This is used for both temporary storage of images while
processing them for classification and for permanent storage of datasets 
constructed while training and testing models. It contains a global store of 
images with associated metadata, from which datasets can be constructed as 
subsets.

## Importing Data
To import data, use the `import_images` function in `pipeline/store.py`. It can
import images both with and without associated class labels (defaulting to an
unlabeled class label, defined by the constant `DEFAULT_CLASS` in 
`pipeline/store.py`). The full list of class labels is given as an int-valued 
dict `CLASSES` in `pipeline/store.py`, which is used to convert labels to 
integers for dataset creation. Images can be imported either by file path on 
disk or by URL from the web, and are imported into the global data store with a 
record of their label.

## Processing Data - Conversions
Images in the global store can have a series of image conversions applied. The
store metadata tracks what conversions have been applied to what images. The 
list of available conversions is given in the `CONVERSIONS` function-valued dict
in `pipeline/conversions.py`, and instructions on adding more conversions are
also specified there. Note that adding more conversions requires destroying and
re-creating the data store or manually adding a metadata column to the metadata
CSV at this time.

Conversions can be applied via the `convert_images` function of 
`pipeline/store.py`, which takes in the image file names (which can be read 
from the metadata CSV of the store) and a list of conversions to apply.

## Creating Datasets

Datasets are created by taking subsets of the current image store and ensuring
each image in the dataset has the same set of conversions applied. To create a
dataset, use the `new_dataset` function in `pipeline/dataset.py`, and provide
the desired conversions that the dataset should have. This will copy the images
to the new dataset and apply conversions to these copies if they have not been
applied in the global store. Datasets can also be created using images directly
from disk instead of from the global store.

Delete a dataset with the `delete_dataset` function.

## Processing Data - Transforms

Transforms are operations applied to images during the conversion into model
training data. They are temporary, applied only to the image data loaded into
memory and saved as training data. Transforms are structured in the same way as
conversions and are listed in `TRANSFORMS` of `pipeline/transforms.py`. One 
notable difference is that adding transforms can be done without resetting or 
altering the global store.

Training data is made with the `make_data` function of `pipeline/dataset.py`,
where transforms are provided along with the dataset. The `bundled` parameter
determines whether or not all chart classes should be treated as one class
(i.e., whether the task is classifying charts from non-charts or also 
classifying different types of charts from each other).

The `get_process` function of `pipeline/dataset.py` allows one to fetch the 
process data of a dataset: a JSON object indicating the conversions, latest
transformations, and latest bundled parameter of the dataset.

## Modelling

There are 4 functions provided for modelling data in `modelling.py`.

The `train_and_save` function takes in a `scikit-learn` classifier, a dataset, 
a list of transforms, and a bundled flag, and trains a model for the given 
dataset and options, saving it with the dataset. It is meant to be paired with
the `load_and_predict` function, which takes a dataset containing a model and a
dataset containing images and classifies the images of the latter dataset using
the model from the prior dataset. Because the `load_and_predict` function must
apply transforms to the data contained in the test dataset, which overwrites 
the process data without overwriting any models associated with the test 
dataset, it is recommended to create datasets explicitly for model training or 
for testing.

The `export_model` function trains a model and exports it to the target 
directory, along with the process data. This does not create the same file type
as the `train_and_save` function. It is meant to be used with the 
`end_to_end_prediction` function, which takes in an exported model and a list 
of image paths, creates a temporary dataset out of the images, classifies the 
processed images, and deletes the temporary dataset before returning the 
classifications.
