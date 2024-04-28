
import torch
import torchvision
from torch import nn
from helper import setAllSeeds

def getEffNetModel(seed,numClasses):
  setAllSeeds(seed)
  effNetWeights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  effNetTransforms = effNetWeights.transforms()
  effNet = torchvision.models.efficientnet_b2(weights=effNetWeights)
  for param in effNet.parameters():
    param.requires_grad = False
  effNet.classifier = nn.Sequential(
    nn.Dropout(p=0.3,inplace=True),
    nn.Linear(1408,numClasses,bias=True)
  )
  return effNet,effNetTransforms
