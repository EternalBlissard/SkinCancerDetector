
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import random
import zipfile
from pathlib import Path
import requests

def setAllSeeds(seed):
  os.environ['MY_GLOBAL_SEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  
def dataDownloader(src,dest):
  downloadPath = Path("downloadedData/")/dest

  if(downloadPath.is_dir()):
    print(f"{downloadPath} directory already exists, skipping downloading procedure")
  else:
    print(f"{downloadPath} directory doesn't already exists, starting downloading procedure")
    downloadPath.mkdir(parents=True,exist_ok=True)
    target = Path(src).name
    with open(Path("downloadedData/")/target,"wb") as f:
      requested = requests.get(src)
      print(f"Downloading {target} from {src}")
      f.write(requested.content)
    
    with zipfile.ZipFile(Path("downloadedData/")/target,"r") as zipRef:
      print(f"Unzipping the data")
      zipRef.extractall(downloadPath)
      os.remove(Path("downloadedData/")/target)
  return downloadPath
