import tensorflow as tf
import numpy as np
from imageio import imread   
from scipy import misc
from PIL import Image
import base64

# import aiohttp
# import asyncio
# from pathlib import Path

# path = Path(__file__).parent

# export_file_url = 'https://drive.google.com/uc?export=download&confirm=xtKg&id=10HnoWxCaeaVgDVZStj19_X_OO7L-YxfC'
# export_file_name = 'test'

# async def download_file(url, dest):
#     if dest.exists(): return
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             data = await response.read()
#             with open(dest, 'wb') as f:
#                 f.write(data)

# async def get_file():
#     await download_file(export_file_url, path / export_file_name)
#     print('ok')

# loop = asyncio.get_event_loop()
# tasks = [asyncio.ensure_future(get_file())]
# learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
# loop.close()

def receive_data(request):
    decoded_image = decode_data(request)
    processed_image = process_data(decoded_image)
    prediction = predict_cancer(processed_image)

    return prediction
   
def decode_data(request):
    b64_encoded = request.decode('utf-8') # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
    # https://stackoverflow.com/questions/57318892/convert-base64-encoded-image-to-a-numpy-array
    b64_decoded = base64.b64decode(b64_encoded[9:-2]) # spliced to get the actual image string
    decoded_image = np.array(imread(b64_decoded))

    # https://stackoverflow.com/questions/56204985/how-to-fix-scipy-misc-has-no-attribute-imresize/56205147
    image = Image.fromarray(decoded_image).resize((600,450))
    # image.show()
    decoded_image = np.array(image) # https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
    
    return decoded_image

def process_data(decoded_image):
    # change dimension
    # https://stackoverflow.com/questions/41563720/error-when-checking-model-input-expected-convolution2d-input-1-to-have-4-dimens (Danny Wang)
    processed_image = np.expand_dims(decoded_image, axis=0) 

    return processed_image

def predict_cancer(processed_image):
    # load model
    model = tf.keras.models.load_model('medical_imaging_model')
    predictions = model.predict(processed_image)[0]
    predictions = [float(i) for i in predictions] # https://stackoverflow.com/questions/1614236/in-python-how-do-i-convert-all-of-the-items-in-a-list-to-floats
    cancerNames = ["Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses", "Actinic keratoses and intraepithelial carcinoma / Bowen's diseases", "Basal cell carcinoma", "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage"]


    # https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/
    cancerMappings = dict(zip(cancerNames, predictions))
    predicted_cancer_index = np.argmax(predictions)
    predicted_cancer = {cancerNames[predicted_cancer_index]: float(predictions[predicted_cancer_index])}
    cancerMappings.update({'prediction': predicted_cancer}) # https://www.geeksforgeeks.org/python-merging-two-dictionaries/
    
    return cancerMappings