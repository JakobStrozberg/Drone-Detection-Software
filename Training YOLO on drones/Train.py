#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:06:41 2023

@author: jakobstrozberg
"""

# %% Training YOLO on drone data set

from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')  # load a pretrained model (recommended for training) #nano limited buy 416

# Train the model
results = model.train(data='/Users/jakobstrozberg/Documents/GitHub/SystemsEngineeringDroneDetection/Training YOLO on drones/data.yaml', epochs=60, imgsz=960, device=0)

# Export the model
model.export()

