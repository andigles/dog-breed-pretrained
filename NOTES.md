# Notes for Dog Breed Image Clasification Project

- Entry point (`check_images.py`)
  Command-line interface that accepts:

  - `--dir` : folder of images to classify

  - `--arch` : model architecture (resnet, vgg, or alexnet)

  - `--dogfile` : path to a text file with dog breeds (one per line)

    Example:
    python check_images.py --dir data/sample/pet_images --arch vgg --dogfile data/dognames.txt
    Output: prints a summary of results to the console. Some versions support optional flags to print misclassified dogs and misclassified breeds (see -h).

- Classifier loop (`classify_images.py`).

  Uses `classifier(image_path, model_name)` to get a classifier label for each image, compares it with the true label parsed from the filename, and stores:

  - the classifier label, and

  - whether it matches the true label (0/1)
  
    …inside `results_dic` (keyed by filename).

- Model helper (`classifier.py`)

  Loads a pretrained network and returns a human-readable label string for an image (lowercased/cleaned). The legacy version used a text mapping file; a modern version uses TorchVision’s built-in label metadata.
  