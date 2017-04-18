import io
import os

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
vision_client = vision.Client()

# write into log file
log_file = './googlevision.log'
log = open(log_file, "w")

# path of the image directory
file_dir = '../dataset_tool/python/testimg_datala/'
for root, dirs, files in os.walk(file_dir):
    for file in files:
        
        file_path = os.path.join(root, file)
        log.write("\n" + file + "\n")

        # loop through all the image files
        with io.open(file_path, 'rb') as image_file:
            # read in image file
            content = image_file.read()
            # talk to google client api
            image = vision_client.image(content=content)

            # detect
            labels = image.detect_labels()
            
            # loop through the replied data
            for label in labels:
                # detected object description
                log.write(label.description + " : ")
                # confident rate
                log.write(str(label.score) + "\n")
                # we only need pedestrian detection
                # setup threashold as 50%
                #if(label.description=='pedestrian' and label.score > 0.5):
                if(label.description=='pedestrian'):
                    print("Detected in ", file, label.score)    

log.close()
