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

4.  Copy or create a symbolink link to your dataset. e.g.:

	``` bash
	cd datasets_dir
	ln -s <your path>/patchcamelyon/ patchcamelyon
	```
	The app used to create the dataset was [this one](https://github.com/giussepi/BACH_ICIAR_2018).
	However, the important thing to keep in mind is that the app works with numpy arrays. We saved
	our dataset as JSON files so you can take a look the loader [here](https://github.com/giussepi/spatial-pyramid-features/blob/master/utils/datasets/patchcamelyon.py). About the format of the arrays:

	1. The features must be in the rows and the number of columns must represent the number of samples.

	2. The labels must be integers represented by the rows and the columns must represent the number of samples. For instance, in the following image we can see a tiny label matrix of three labels
		and four samples:

		<img src="doc_images/example_label_matrix.png" width="50%"/>

5. Copy the settings template and configure it properly.

	``` bash
	cp settings.py.template settings.py
	```

## Usage

### Create spatial pyramid features
``` python
from utils.datasets.patchcamelyon import DBhandler

spf = SpatialPyramidFeatures(DBhandler)
spf.create_codebook()
spf.create_spatial_pyramid_features()
```


### Load spatial pyramid features
``` python

from utils.datasets.patchcamelyon import FeatsHandler

train_feats, train_labels, test_feats, test_labels = FeatsHandler()()
```
