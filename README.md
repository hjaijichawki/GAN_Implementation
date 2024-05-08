*GAN Implementation for Image Generation*
--------------------------------------

**Introduction**
------------------------

GANs, or Generative Adversarial Networks, are a class of deep learning models that consist of two neural networks, a generator, and a discriminator, which are trained simultaneously through adversarial training. The goal of GANs is to generate realistic data, often images, that is indistinguishable from real data.The generator is a neural network that takes random noise as input and generates synthetic data, attempting to mimic the distribution of the real data.The discriminator is another neural network that evaluates whether an input is real (from the true data distribution) or fake (generated by the generator). Its objective is to correctly classify real and generated samples.The generator and discriminator are trained concurrently in a competitive manner. The generator aims to produce increasingly realistic samples to fool the discriminator, while the discriminator strives to become more accurate in distinguishing real from fake samples.

![GAN](https://github.com/hjaijichawki/GAN_Implementation/assets/116977931/3a97f649-f32f-4a97-9420-5db19cb2efc1)





**Dataset**
----------------


You find here the link for the dataset.
[Brain Tumor Detection MRI Dataset](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri?fbclid=IwAR0vZxyXazz_k64eRmOP7X-ltavMYQl5dS28QSskOXV2mEnMdEXjfhLiCPA)


**Code Structure**
---------------------------


* ***Directory_creation.py:*** Contains the code for creating directories for the training process.
  
*  **dataset_extract.py:** Contains the code for dataset extraction from a zip file.

* ***Data_Preparation.py:*** Contains the code for splitting data.
  
*  **Display.py:** Contains the code for displaying images for the dataset.

* ***Image_generation.py:*** Contains the code for generating images. 
* ***main.py:*** Contains the main code to run it. 
* ***models.py:*** Contains the code for the generator and the discriminator network. 
* ***Plots.py:*** Contains the code for displaying plots. 
* ***Show_generated_images.py:*** Contains the code for displaying generated images. 
* ***Transformations.py:*** Contains the code for applying transformations on images. 
  


**Running Code**
-------------------

* Download the dataset by following the instructions below:

    Download kaggle's beta API

    Execute the following commands
    ```
        install kaggle.json 
        !pip install -q kaggle
        !mkdir ~/.kaggle
        !cp kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json
        !kaggle datasets download -d abhranta/brain-tumor-detection-mri
* Execute the command `pip install -r requirements.txt` 
* Use the function `create_dir` in `Directory_creation.py` to create the training directory.
* Run the code `dataset_extract.py` to extract the dataset from a zip file
* Use the function `prepare` in `Data_Preparation.py` to split dataset.
* Use the function `display` in `Display.py` to display some images from the dataset.
* Run the code `main.py` 
* Run the following script
  ````
  !apt-get install tree
  !mkdir generated_images 
  !tree -d
* Use the function `generate_images` in `Image_generation.py` to generate new synthetic images.
* Use the function `show_generated_images` in `Show_generated_images.py` to display the generated images.



**Contribution**
------------------------

Contributions are welcome! Feel free to open issues or create pull requests to enhance the functionality or address any improvements.






