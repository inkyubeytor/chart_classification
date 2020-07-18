# chart_classification

A chart classification pipeline.

TODO:
* Given list of URLs or paths, add images to data store
  * If a list of classes is given, add with those classes
* Given a list of image names from data store and list of conversions, apply
  conversions to images in data store (destructively)
* Given list of image names from data store and list of conversions, make a new 
  dataset with those converted images
  * Assert those images have no other conversions applied
  * Copy the images, then destructively convert them
  * Record the set of conversions in process.json
* Given a dataset, convert the images into model data npy files
  * Record the set of transforms in process.json

* Train/Classify
  * Load training data
  * Train and save model
  * Load model
  * Predict with model