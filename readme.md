# image extraction from zara webpage

produce pages:
https://www.zara.com/us/en/dtrt-jckt-13-p04164921.html?v1=405265154&v2=2467336


## test 1 results:
we can get the image url from the webpage for zara. 
bottleneck: manually check the format of the image url from the network console


## test 2 logic: can we extract from a different webpage? bloomingdales

## test 3
if we were to extract from different webpages, what next steps? what do you with the list of image urls?

# next step:
1. we can use the images to train a model to extract the image url from the webpage
2. we can use the images to train a model to classify the image
3. we can use the images to train a model to segment the image
4. we can use the images to train a model to generate the image

but for training purposes, we already have databases of images publicly available. why not use them?
exactly.

we are training for a image feature extraction task that can be used for retrieval tasks
such as image search and recommendation. 
1. create a large database of image urls from all the branded stores.
2. given a new image, find a similar image from the database.
3. given a new image, find the product description from the database (including the price, and url)

run critical tests manually:
1. given a new image, can you find a similar image from the database?
2. how do you define similar?