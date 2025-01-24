import os
import cv2
import json
import math
import base64
from PIL import Image

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def video2frame(video_path, save_dir=None, num_frames=4):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error opening video file")
    return []

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  interval = total_frames // num_frames

  idx = 0
  saved = []
  while True:
    ret, frame = cap.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
        
    if idx % interval == 0 and len(saved) < num_frames:
      path = os.path.join(save_dir, f"{idx}.jpg")

      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img_pil = Image.fromarray(rgb_frame)

      cv2.imwrite(path, frame)
      # print(f'Saved frame {idx} at {path}')
      saved.append(path)

    if len(saved) >= num_frames:
      break

    idx += 1

  cap.release()
  print('Done saving frames.')
  return saved

def read_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def read_json(json_url):
    with open(json_url, 'r') as file:
        data = json.load(file)
    return data

def write_json(data, save_url, indent=4):
    with open(save_url, 'w') as file:
        json.dump(data, file, indent=indent)
    return save_url

def read_task(task_json):
    task_list = []
    for x in task_json:
        task_list.append(x['text'])
    task_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(task_list)])
    return task_str

def print_score(procedure_eval):
    num_seq = len(procedure_eval)
    score_dict = {'vis': 0, 'txt': 0, 'vis_txt': 0}
    for x in procedure_eval:
        for mode in ['vis', 'txt', 'vis_txt']:
            score_dict[mode] += x[mode]['score']

    for mode in ['vis', 'txt', 'vis_txt']:
        score_dict[mode] /= num_seq
    return score_dict

def point_to_rect(point, pred): 
    # point: [x,y]
    # rect: [[x1,y1,x2,y2]]
    x, y = point
    x1, y1, x2, y2 = pred[0]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    inside = x1 <= x <= x2 and y1 <= y <= y2
    return distance, inside

def point_to_point(point, pred):
    # point: [x,y]
    x, y = point
    center_x, center_y = pred
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance