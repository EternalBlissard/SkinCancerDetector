### Imports for Modules ### 
import gradio as gr
import os
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer

### Functional Imports
from predictor import predictionMaker

exampleList = [["examples/" + example] for example in os.listdir("examples")]

title = "Detecting Skin Cancer early for a Better tommorrow"
description = "An EfficientNetB2 feature extractor computer vision model to classify images into Skin Cancer Types: Actinic Keratosis, Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Pigmented Benign Keratosis, Seborrheic Keratosis, Squamous Cell Carcinoma, Vascular Lesion"
article = "Created by [Eternal Bliassard](https://github.com/EternalBlissard)."

# Create the Gradio demo
demo = gr.Interface(fn=predictionMaker, 
                    inputs=[gr.Image(type="pil")], 
                    outputs=[gr.Label(num_top_classes=2, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=exampleList, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch() 




