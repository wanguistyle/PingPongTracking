import os
from dotenv import load_dotenv
from roboflow import Roboflow


load_dotenv() 
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key) 
project = rf.workspace("Pingpong").project("ball_detection")
version = project.version(2)
dataset = version.download("yolov11")

print("Téléchargement terminé !")

