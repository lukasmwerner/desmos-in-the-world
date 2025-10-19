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
    print(eqn)
    print(type(eqn))
    tags_list[tag_number] = eqn
    return eqn

if __name__ == "__main__":
    load_dotenv()
    client = make_gemini_client()
    tags_list = [None for i in range(200,399)]
    print(gemini(client, tags_list, 100, "image.png"))
    print(gemini(client, tags_list, 1, "image copy.png"))
    print(tags_list)