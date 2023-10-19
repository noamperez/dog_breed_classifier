# A Dog Breed Classifier :dog:
## Intro
You can explore the [model](https://huggingface.co/spaces/noamperez/dog_breed_classifier) using your own examples.

A self-initiated project to create a website for dog breed detection.
We can outline the sections of this project as follows:

1. Data download and local directory reorganization (combining two separate datasets).
2. Baseline model development.
3. Enhancement of the baseline model.
4. Deploy the model.

## Data download and local directory reorganization
I acquired two primary datasets: [The Stanford Dogs Dataset](https://www.tensorflow.org/datasets/catalog/stanford_dogs) and [The Oxford-IIIT Pet Dataset ](https://www.robots.ox.ac.uk/~vgg/data/pets/).
The subsequent steps involved:
- Creating a dedicated folder for each dog breed.
- Populating these breed-specific folders with their respective images, ensuring files were appropriately renamed.
- Eliminating duplicate images from both datasets.

## Baseline model development
In order to train the model, I employed the fastai framework. Here's the code I used:
```python
dogs = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label, 
    item_tfms=Resize(224))
```
Our baseline model was pretrained resnet18
```python
dls = dogs.dataloaders(path)
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(5)
```
The results are as follows
| epoch | train_loss | valid_loss | error_rate | time  |
|-------|------------|------------|------------|-------|
| 0     | 1.455887   | 1.050625   | 0.307084   | 01:10 |
| 1     | 1.369547   | 1.045225   | 0.303857   | 01:06 |
| 2     | 1.068617   | 0.942868   | 0.276263   | 01:06 |
| 3     | 0.904221   | 0.865005   | 0.257060   | 01:07 |
| 4     | 0.819382   | 0.856869   | 0.254639   | 01:09 |

Then, I saved the model in a pickle file
```python
learn.export('breed_detector.pkl')
```

## Enhancement of the baseline model :chart_with_upwards_trend:
In order to improve our performance, we used several methods.
The first method is using data augmantaion
```python
dogs = dogs.new(
    item_tfms=RandomResizedCrop(128, min_scale=0.5),
    batch_tfms=aug_transforms())
```
Second, we changed the model to [convnext_tiny_in22k](https://huggingface.co/timm/convnext_tiny.fb_in22k). I selected this model because it outperforms ResNet18, yet it's a more lightweight version of a larger model that I couldn't run on my personal laptop. Furthermore, when deploying the model, a smaller model generally runs faster.
```python
dls = dogs.dataloaders(path)
learn = vision_learner(dls, 'convnext_tiny_in22k', metrics=error_rate)
learn.fine_tune(5)
```
The results are as follows

| epoch | train_loss | valid_loss | error_rate | time |
|-------|------------|------------|------------|------|
| 0     | 0.768896   | 0.633528   | 0.192190   | 11:25|
| 1     | 0.737943   | 0.609681   | 0.181217   | 11:24|
| 2     | 0.585935   | 0.529555   | 0.156366   | 11:26|
| 3     | 0.428763   | 0.471445   | 0.141197   | 11:20|
| 4     | 0.375453   | 0.450840   | 0.138938   | 11:21|

When experimenting with other data augmentation methods or alternative models, it didn't yield significant improvements.
Then, I saved the model in a pickle file
```python
learn.export('breed_detector_2.pkl')
```

## Deploy the model :package:
Now, as we have a functional model, it's time to build a website for dog breed detection. To achieve this, we've leveraged the [Hugging Face platform](https://huggingface.co/). Here's the roadmap for the implementation:

First, we will load the pickle file we saved earlier.
```python
learn = load_learner('breed_detector_2.pkl')
labels = learn.dls.vocab
```

Then, we use huggingface and [gradio](https://www.gradio.app/) to deploy the model into a website.
```python
gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(192, 192)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch(share=True)
```







