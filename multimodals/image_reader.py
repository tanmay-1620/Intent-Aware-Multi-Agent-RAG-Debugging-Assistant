import pytesseract
from PIL import Image

# Change path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()

    except Exception as e:
        return f"Error reading image: {e}"


if __name__ == "__main__":
    path = input("Enter image path: ")
    result = extract_text_from_image(path)
    print("\nExtracted Text:\n")
    print(result)