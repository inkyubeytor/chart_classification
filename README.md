# chart_classification


TODO:
* Get an image from a source
  * Via URL
  * From on disk

* Convert a raw image to an input
  * Convert to png
  * Scale to 300H x 400W
  * Convert to grayscale
  * Convert to numpy array
  * Scale to be in the range [0, 1]

* Train/Classify
  * Load training data
  * Train and save model
  * Load model
  * Predict with model