import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAzgDIDix6OQoiSxBUjM5Fp3SEBCJpvdQk")

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)