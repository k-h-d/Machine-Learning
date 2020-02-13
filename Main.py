import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import datetime
import glob
import random

model=tf.keras.applications.vgg16.VGG16()
model.summary()