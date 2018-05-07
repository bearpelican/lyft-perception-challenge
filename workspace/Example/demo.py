import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

for rgb_frame in video:
	
    # Grab red channel	
	red = rgb_frame[:,:,0]    
    # Look for red cars :)
	binary_car_result = np.where(red>250,1,0).astype('uint8')
    
    # Look for road :)
	binary_road_result = binary_car_result = np.where(red<20,1,0).astype('uint8')

	answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
    # Increment frame
	frame+=1

# Print output in proper json format
print (json.dumps(answer_key))