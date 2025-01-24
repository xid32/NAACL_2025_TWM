import time
import os
import json
import imageio
import requests
import argparse
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from utils import encode_image

import openai
from openai import OpenAI

class MovieSeq:
    def __init__(self,
                    model="gpt-4o", api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail
        if system_text is None:
            self.system_text = """
        You will be provided with the following inputs:
        1. A sequence of photos of characters along with their names.
        2. Keyframes from a video clip and the corresponding dialogues, each associated with a speaker ID.
        
        Your task is to analyze and associate these inputs, understand the context of the video, and respond to the user's needs accordingly.
        """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

    def get_response(self, char_bank, frame_list, diag_list, 
                                query,
                                resize=None, temperature=0, detail="auto"):
        messages = [{
            "role": "system", 
            "content": [{"type": "text", "text": self.system_text,},]
            }]
            
        for char_name, char_url in char_bank.items():
            char_image = encode_image(char_url)
            messages.append({
                "role": "user",
                "content": [
                    f"This is the photo of {char_name}.",
                    {'image': char_image},
                ],
            })

        assert len(diag_list) == len(frame_list)
        for frame_i, diag_i in zip(frame_list, diag_list):
            frame_image = encode_image(frame_i)
            messages.append({
                "role": "user",
                "content": [
                    {'image': frame_image},
                    f"{diag_i}.",
                ],
            })

        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": query,},]
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        
        response = self.client.chat.completions.create(**params)
        json_string = response.json()
        json_object = json.loads(json_string)
        content = json_object['choices'][0]['message']['content']
        return content