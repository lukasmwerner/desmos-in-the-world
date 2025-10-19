import os
from google import genai
from dotenv import load_dotenv
from PIL import Image
import sympy


def make_gemini_client():
    client = genai.Client(api_key= os.getenv("GEMINI_API_KEY"))
    return client
  

def gemini(client, tags_list, tag_number, image):
    img = Image.fromarray(image)

    eqn = sympy.sympify(client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
        img,
        'ONLY PROVIDE THE SYMPY STRING REQUESTED, DO NOT INCLUDE QUATATIONS. THIS IS NOT A PYTHON SCRIPT DO NOT WRITE ANY PYTHON EVER. Given this image, generate a string of the equation provided in sympy format in order for sympy to underestand in the context of a function call for evaluating a mathematical. Example: (sin(x) - 2*cos(y)**2 + 3*tan(z)**3)**20)'
        ]
    ).text)
    tags_list[tag_number] = eqn
    return eqn

if __name__ == "__main__":
    load_dotenv()
    client = make_gemini_client()
    tags_dict = {}
    for i in range(200, 400):
        tags_dict[i] = None
    print(tags_dict)