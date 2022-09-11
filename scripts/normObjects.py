import os
import json
import shutil
import numpy as np
import cv2
from tqdm import tqdm

MASKS_FOLDER = "D:\Cleansea\cleansea_dataset\Dataset\Masks_new_trans"
DISPLACED_MASKS = "D:\Cleansea\cleansea_dataset\Dataset\Objetos"
dataset = os.listdir(MASKS_FOLDER)
for dir in tqdm(dataset, "Procesando Objetos"):
    dir_path = os.path.join(MASKS_FOLDER,dir)
    files = os.listdir(dir_path)
    for file in tqdm(files,f"Procesando Objetos {dir}", leave= False):
        if file.endswith(".json"):
            f = open(os.path.join(dir_path,file))
            data = json.load(f)
            linked_img = data["imagePath"]
            if len(data["shapes"]) != 0:
                obj=0
                s = data["shapes"]
                img = cv2.imread(os.path.join(dir_path,linked_img))
                label = s["label"]
                if not os.path.exists(os.path.join(DISPLACED_MASKS,label)):
                    os.makedirs(os.path.join(DISPLACED_MASKS,label))
                pts = np.array(s["points"],dtype=np.int32)
                max_x = 0
                min_x = data["imageWidth"]
                max_y = 0
                min_y = data["imageHeight"]

                for x,y in pts:
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

                desp_x = min_x
                desp_y = min_y
                desp_pts = []
                mask_w = max_x - min_x
                mask_h = max_y - min_y

                for x,y in s["points"]:
                    new_x = x-desp_x
                    new_y = y-desp_y
                    new_pt = [float(new_x),float(new_y)]
                    desp_pts.append(new_pt)
                desp_pts_np = np.array(desp_pts,dtype=np.int32)
                
                mask = np.zeros((img.shape[0], img.shape[1]))
                mask_desp = np.zeros((img.shape[0], img.shape[1]))
                cv2.drawContours(mask, [pts], -1, (1), thickness=-1)
                cv2.drawContours(mask_desp, [desp_pts_np], -1, (1), thickness=-1)
                mask = mask.astype(np.bool)
                mask_desp = mask_desp.astype(np.bool)

                out = np.zeros_like(img)
                out[mask_desp] = img[mask]
                out = out[:mask_h,:mask_w]

                #cv2.imshow(f'{linked_img}', out)
                #cv2.waitKey()
                #cv2.destroyAllWindows()
                
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

                name = os.path.splitext(linked_img)[0] + "_"+str(obj) + ".png"
                json_name = name.replace("png","json")
                cv2.imwrite(os.path.join(DISPLACED_MASKS,label,name),dst)
                
                new_data = {}
                for field in data:
                    if field == "shapes":
                        new_data[field] = [s]
                        new_data[field][0]["points"] = desp_pts
                    elif field == "imagePath":
                        f_data = name
                        new_data[field] = f_data
                    elif field =="imageData":
                        f_data = ""
                        new_data[field] = f_data
                    elif field =="imageHeight":
                        f_data = int(mask_h)
                        new_data[field] = f_data
                    elif field =="imageWidth":
                        f_data = int(mask_w)
                        new_data[field] = f_data
                    else:
                        f_data = data[field]
                        new_data[field] = f_data

                json.dump(new_data,open(os.path.join(DISPLACED_MASKS,label,json_name),"w"),indent=2,separators=(", ",": "),sort_keys=False)

                #shutil.copy(os.path.join(ANNOTATIONS_FOLDER,file),os.path.join(MASKS_FOLDER,label,file))
                obj +=1
            