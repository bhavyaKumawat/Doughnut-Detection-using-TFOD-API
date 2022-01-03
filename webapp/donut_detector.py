import urllib.request
from PIL import Image
from flask import request
import numpy as np
import json
import os

def detect_donuts(filename): 
 
    img = request.files['image']
    img = Image.open(img) 
    img = img.resize(size=(400, 400))
    img = np.array(img) 
    
            
    # Request data goes here
    data = { "data": img.tolist()
    }
    
    
    body = str.encode(json.dumps(data))
    
    url = 'http://donut-detection-endpoint.centralindia.azurecontainer.io/score'
    api_key = '0IgOuJCMnxq5JAgr7Olw78YPaXSQWLa8' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    

    req = urllib.request.Request(url, body, headers)

    try:    
        response = urllib.request.urlopen(req)
        result = response.read()
        predictions = json.loads(result)
        
        array = np.array(predictions["output_image"], dtype=np.uint8)
        new_image = Image.fromarray(array)
        
        
        if not os.path.exists(os.path.join('static', 'outputs')):
            os.mkdir(os.path.join('static', 'outputs'))
            
        image_path = os.path.join('static', 'outputs', filename)
        new_image.save(image_path)
	            
        return predictions["count"]
        
        

    except urllib.error.HTTPError as error:
        return "The request failed with status code: " + str(error.code)
        
        
        









    


    
    
