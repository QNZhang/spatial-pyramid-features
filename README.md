# SPATIAL PYRAMID FEATURES

Python implementation of the [spatial pyramid features](https://ieeexplore.ieee.org/document/1641019) used on [2013 Label Consistent K-SVD: Learning a Discriminative Dictionary for Recognition](https://ieeexplore.ieee.org/abstract/document/6516503).


## INSTALLATION


1. Create virtual environment (recommended) and activate it

2. Install all the dependencies

	```bash
	pip install -r requirements
	```
3. Create the datasets directory

	```bash
	mkdir datasets_dir
	```

4.  Copy or create a symbolic link to your dataset. e.g.:

	``` bash
	cd datasets_dir
	ln -s <your path>/patchcamelyon/ patchcamelyon
	```
	The app used to create the dataset was [this one](https://github.com/giussepi/BACH_ICIAR_2018).
	However, the important thing to keep in mind is that the app works with numpy arrays. We saved
	our dataset as JSON files so you can take a look at the loader [here](https://github.com/giussepi/spatial-pyramid-features/blob/master/utils/datasets/handlers.py). About the format of the arrays:

	1. You must provide a training and testing JSON files. These files must contain a JSON serialized python dictionary with to keys: 'codes' and 'labels'.

	2. When turning the 'codes' values to a 2D numpy array, this array must have the shape `(num features, num samples)`. Thus, the each column must refer to a sample.

		``` python
			# OPTION 1: Full datastet to be loaded in memory
  			'codes': [[values_row_1], [values_row_2], [values_row_3], ...]

			# The easiest way to create this option using numpy is:
			'codes': [[mydata[i].tolist() for i in range(mydata.shape[0])] ]

			# OPTION 2: Images paths (memory efficient)
			'codes': ['img_path_1', 'img_path_2', 'img_path_3', ...]
		```

		For the OPTION 1 you can get some inspiration/clarification by looking at the class [BaseDatasetCreator](https://github.com/giussepi/BACH_ICIAR_2018/blob/master/utils/datasets/bach.py) and its derivates RawImages and RandomFaces. Specifically, look for the methods: 'process_data, format_for_LC_KSVD, create_datasets_for_LC_KSVD'.

	3. For the labels (must be integers) you also have two options:
		1. The labels must be represented by the rows and the columns must represent the number of samples. For instance, in the following image we can see a tiny label matrix of three labels
		and four samples:

	       <img src="doc_images/example_label_matrix.png" width="50%"/>

		2. The labels can be a 1D array, holding the values of the labels, e.g.: The follwing array is the 1D representation of the labels example presented previously.

	       ``` python
			   [1, 0, 2, 2]
		   ```

5. Copy the settings template and configure it properly.

	``` bash
	cp settings.py.template settings.py
	```

## Important notes

### Grey-scale images
If the image dataset does not have the grey-scale PIL format ('L', ), then each sample will be automatically converted (in memory) before any processing. This behaviour is defined at:

    `utils/datasets/items.py -> LazyDatasetItems.get_sample`

### Quick runs/tests while tweaking configuration and/or code
If you want to perform quick tests by using only a fraction of your dataset just go to your `settings.py` file and set `QUICK_TESTS` with the number of samples you want to work with, e.g.:

```python
	QUICK_TESTS = 1001
```

Note that only the first 1001 samples will be used; it won't get a fixed number of samples per class. This feature is just to do quick test when changing the configuration (mainly to verify the processing time and RAM usage) or modifying the code (to verify that it still works)


### Setting up 15-Scene dataset (start playing with the code!)

Running the application using 15-Secene dataset is very simple:

1. Download the 15-Scene dataset from any of these locations (or just google it)

	1. [Ready to go option](https://github.com/TrungTVo/spatial-pyramid-matching-scene-recognition/tree/master/data). This one is ready to use.

	2. [Download from figshare](https://figshare.com/articles/15-Scene_Image_Dataset/7007177)

	3. [Download from Kaggle](https://www.kaggle.com/zaiyankhan/15scene-dataset)

2. Split the dataset into train and test folders. If you are using the ready to go option this process is already done.

3. Create a folder called `scene_data` in your project root and place your train and test folders there.

4. Create the required dataset JSON files. Let us start working with 100 samples per class. Thus, the following code must be executed from the main.py file.
	``` python
	from utils.datasets.handlers import LazyDBHandler
	from utils.utils import create_15_scene_json_files

	create_15_scene_json_files(100)
	train_feats, train_labels, test_feats, test_labels = LazyDBHandler()()
	print(train_feats.num_samples, train_labels.shape, test_feats.num_samples, test_labels.shape)
	```

5. Create your codebook and spatial pyramid features
   ``` python
   from core.feature_extractors import SpatialPyramidFeatures
   from utils.datasets.handlers import LazyDBHandler

   spf = SpatialPyramidFeatures(LazyDBHandler)
   spf.create_codebook()
   spf.create_spatial_pyramid_features()
   ```

6. Evaluate the spatial pyramid features using Linear Support Vector Classification
   ``` python
   from utils.utils import FeaturesEvaluator

   FeaturesEvaluator.apply_linear_svc()
   FeaturesEvaluator.find_best_regularization_parameter()
   ```

	Using the default configuration you should get an accuracy close to 68.5%

## Usage

### Load all train/test datasets in memory

The JSON files must contain all the image values/features so they can be loaded into numpy arrays with shape `(features, samples)`.

The original width and size of the images must be set on your `settings.py` files:

``` python
IMAGE_WIDTH = 32  # original image with
IMAGE_HEIGHT = 32  # original image height
```

Then you are ready to start using the handler.

```python
from utils.datasets.handlers import InMemoryDBHandler

train_feats, train_labels, test_feats, test_labels = InMemoryDBHandler()()
```

### Load only image paths and load them only when necessary

The JSON files must have paths to the images which must be accessible from the project root directory. Thus, you can create a symbolic link or place your dataset directory in the project root directory to enable the image paths from the project root directory.

```python
from utils.datasets.handlers import LazyDBHandler

train_feats, train_labels, test_feats, test_labels = LazyDBHandler()()
```

### Create spatial pyramid features
``` python
from core.feature_extractors import SpatialPyramidFeatures
from utils.datasets.handlers import InMemoryDBHandler, LazyDBHandler

# For datasets completely contained in json files
spf = SpatialPyramidFeatures(InMemoryDBHandler)
# For datasets holding just paths to images in the JSON files
spf = SpatialPyramidFeatures(LazyDBHandler)
#
spf.create_codebook()
spf.create_spatial_pyramid_features()
```

### Load spatial pyramid features
Use this dataset handler to load all the spatial pyramid features created by this application


``` python
from utils.datasets.handlers import FeatsHandler

train_feats, train_labels, test_feats, test_labels = FeatsHandler()()
```

### Create a subset of the spatial pyramid features dataset and load it

Use **`FeatsHandler().create_subsets`** to create subsets of the spatial pyramimd features training dataset considering the provided percentage as the percentage covered by the subset training dataset. Thus, the spatial pyramid features training dataset is splitted into `percentage`% for training and `100-percentage`% for testing.

Then just load your feature subsets using the `FeatsHandler` along with the `percentage` of the training dataset used. E.g.:

``` python
from utils.datasets.handlers import FeatsHandler

# Create a 20% training and 80% testing feature subdatasets
FeatsHandler().create_subsets(percentage=20, verbose=True)

# Load the created features subsets
train_feats, train_labels, test_feats, test_labels = FeatsHandler(percentage=20, verbose=True)()
```
