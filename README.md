# maskrcnn production in GCP. Working code

 ```
from PIL import Image
import io
import requests

img = Image.open("./images/test1.jpg")
url = "http://35.229.23.156:5000/predict/"
files = {}
with io.BytesIO() as output:
    img.save(output, format="png")
    contents = output.getvalue()
    filename = "test"
    files[filename] = contents
response = requests.post(url, files = files)
```
