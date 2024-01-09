import os
import cv2
from visualize_cv2 import display_instances, class_names, random_colors, inference_config, MODEL_DIR
import sys
import mrcnn.model as modellib

args = sys.argv

if(len(args) < 3):
    print("\n ERROR -> Run: python video_demo.py video_file_path model_path\n")
    sys.exit(0)
name=args[1]
MODEL_PATH = args[2]
if(len(args[1]) == 1):
    name = int(args[1])

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=inference_config
)

model.load_weights(MODEL_PATH, by_name=True)


stream = cv2.VideoCapture(name)

size = (
    int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

if not os.path.exists("detection"):
    os.makedirs("detection")
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('detection/debris_detection_v2.avi', codec, 60.0, size)

while True:
    ret, frame = stream.read()
    if not ret:
        print("unable to fetch frame")
        break
    results = model.detect([frame], verbose = 1)

    #Visualize Results
    r = results[0]

    masked_image =  display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    
    output.write(masked_image)
    cv2.imshow("Deteccion_Residuos",masked_image)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

stream.release()
cv2.destroyAllWindows("Deteccion_Residuos")