# PingPongTracking with deeplearning

This file is meant to explain of the files located in the 'roboflow' directory andf the method that was used through all the duration of the project.

## download_model.py

This file was created to download the YOLOv11 model trained on Roboflow using the free API key.
Unfortunately, the free API key doesn't allow to download the weights of the model so I had to use Google collab and their GPU to train again a blank YOLOv11 model with the dataset generated in Roboflow (labelled and sampled images).

## .env

To load my Roboflow api key.

## results_homography.py

