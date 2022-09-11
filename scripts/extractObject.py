import os
import json
import shutil
import numpy as np
import cv2
from tqdm import tqdm

ANNOTATIONS_FOLDER = "D:\Cleansea\cleansea_dataset\Dataset\Images+Annotations"
MASKS_FOLDER = "D:\Cleansea\cleansea_dataset\Dataset\Masks"

dir = os.listdir(ANNOTATIONS_FOLDER)

for file in tqdm(dir,"Imagenes Procesadas"):
    if file.endswith(".json"):
        f = open(os.path.join(ANNOTATIONS_FOLDER, file))
        data = json.load(f)
        linked_img = data["imagePath"]
        if len(data["shapes"]) != 0:
            obj=0
            for s in data["shapes"]:
                img = cv2.imread(os.path.join(ANNOTATIONS_FOLDER,linked_img))
                
                label = s["label"]
                if not os.path.exists(os.path.join(MASKS_FOLDER,label)):
                    os.makedirs(os.path.join(MASKS_FOLDER,label))
                pts = np.array(s["points"],dtype=np.int32)
                #print(pts)

                mask = np.zeros((img.shape[0], img.shape[1]))
                cv2.drawContours(mask, [pts], -1, (1), thickness=-1)
                mask = mask.astype(np.bool)

                out = np.zeros_like(img)
                out[mask] = img[mask]

                # Convert image to image gray
                tmp = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                
                # Applying thresholding technique
                _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
                
                # Using cv2.split() to split channels 
                # of coloured image
                b, g, r = cv2.split(out)
                
                # Making list of Red, Green, Blue
                # Channels and alpha
                rgba = [b, g, r, alpha]
                
                # Using cv2.merge() to merge rgba
                # into a coloured/multi-channeled image
                dst = cv2.merge(rgba, 4)

                #cv2.imshow(f'{linked_img}', out)
                #cv2.waitKey()
                #cv2.destroyAllWindows()
                name = os.path.splitext(linked_img)[0] + "_"+str(obj) + ".png"
                json_name = name.replace("png","json")
                cv2.imwrite(os.path.join(MASKS_FOLDER,label,name),dst)
                
                new_data = {}
                for field in data:
                    if field == "shapes":
                        new_data[field] = s
                    elif field == "imagePath":
                        f_data = name
                        new_data[field] = f_data
                    elif field =="imageData":
                        f_data = ""
                        new_data[field] = f_data
                    else:
                        f_data = data[field]
                        new_data[field] = f_data
                        
                json.dump(new_data,open(os.path.join(MASKS_FOLDER,label,json_name),"w"),indent=2,separators=(", ",": "),sort_keys=False)

                obj +=1
            