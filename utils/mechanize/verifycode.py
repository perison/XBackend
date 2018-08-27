from PIL import Image
import pytesseract

def generate(path):
    im = Image.open(path)
    text = pytesseract.image_to_string(im)
    return text

if __name__ == "__main__":
    print generate('/Users/perisonchan/Downloads/2.jpeg')