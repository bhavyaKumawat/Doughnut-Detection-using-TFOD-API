import logging
import azure.functions as func
import urllib.request
from PIL import Image
import numpy as np
import json
import io


def main(blobin: func.InputStream, blobout: func.Out[bytes], bloboutjson :func.Out[str]):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {blobin.name}\n"
                 f"Blob Size: {blobin.length} bytes")

    img = Image.open(blobin) 
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
        count = predictions["count"]

        # Use PIL to create an image from the new array of pixels
        output_image = Image.fromarray(array)
        

    except urllib.error.HTTPError as error:
        logging.info( "The request failed with status code: " + str(error.code))
        


    # Store final composite in a memory stream
    img_byte_arr = io.BytesIO()
    # Convert composite to RGB so we can save as JPEG
    output_image.save(img_byte_arr, format='JPEG')

    # Set blob content from byte array in memory
    blobout.set(img_byte_arr.getvalue())
    bloboutjson.set(str(count))

