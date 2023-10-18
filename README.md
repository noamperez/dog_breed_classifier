# A Dog Breed Classifier :dog:
## Intro
You can explore the [model](https://huggingface.co/spaces/noamperez/dog_breed_classifier) using your own examples.

A self-initiated project to create a website for dog breed detection.
We can outline the sections of this project as follows:

1. Data download and local directory reorganization (combining two separate datasets).
2. Baseline model development.
3. Enhancement of the baseline model.

## Data download and local directory reorganization
I acquired two primary datasets: [The Stanford Dogs Dataset](https://www.tensorflow.org/datasets/catalog/stanford_dogs) and [The Oxford-IIIT Pet Dataset ](https://www.robots.ox.ac.uk/~vgg/data/pets/).
The subsequent steps involved:
- Creating a dedicated folder for each dog breed.
- Populating these breed-specific folders with their respective images, ensuring files were appropriately renamed.
- Eliminating duplicate images from both datasets.

## Baseline model development
In order to train the model, I employed the fastai framework. Here's the code I used:
```
dogs = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label, 
    item_tfms=Resize(224))
```
Our baseline model was pretrained resnet18
```
dls = dogs.dataloaders(path)
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)
```



