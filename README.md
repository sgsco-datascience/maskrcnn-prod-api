# maskrcnn production in AKS barcode cluster. Working code

 ```
from PIL import Image
import io
import requests
import time

img = Image.open("./images/test1.jpg")
url = "http://35.229.23.156:5000/predict/"
files = {}
with io.BytesIO() as output:
    img.save(output, format="png")
    contents = output.getvalue()
    filename = "test"
    files[filename] = contents
    
start = time.time()
response = requests.post(url, files = files)
print(time.time()- start)
```
