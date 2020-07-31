from pipeline.store import init_data_store, import_images, convert_images, \
    CLASSES
from pipeline.dataset import new_dataset
from pipeline.conversions import CONVERSIONS
from pipeline.transforms import TRANSFORMS
from modelling import train_and_save, load_and_predict, export_model, end_to_end_prediction
import os
import pandas as pd
from sklearn.svm import SVC


# Import external data
p = "C:/path/to/importable/data"
df = pd.read_csv(f"{p}/metadata.csv")
images, labels = [], []
for _, row in df.iterrows():
    images.append(f"{p}/images/{row['ID']}.png")
    labels.append(list(CLASSES.keys())[list(CLASSES.values()).index(row['Class'])])

init_data_store()
import_images(images, labels, False)

# Apply conversions
filenames = list(pd.read_csv("data/log.csv")["File"])

convert_images(filenames, list(CONVERSIONS.keys()))


# Make new dataset
filenames = [r["File"] for _, r in pd.read_csv("data/log.csv").iterrows()
             if r["Class"] != "Map"]
dataset = new_dataset(filenames, list(CONVERSIONS.keys()))
print(dataset)

# Train model
train_and_save(SVC(degree=1), dataset, list(TRANSFORMS.keys()), True)
# Load and predict
load_and_predict(dataset, dataset)

# Export model
images = "C:/path/to/images"
image_paths = [f"{images}/{p}" for p in os.listdir(images)]
path = "C:/path/to/export_model.joblib"
export_model(dataset, path)
print(end_to_end_prediction(path, image_paths))
