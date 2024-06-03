from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, maintain_aspect_ratio=False):
    img = Image.open(image_path)

    if maintain_aspect_ratio:
        img.thumbnail((32, 32))  # Modify in place
        new_img = Image.new("RGB", (32, 32))
        new_img.paste(img, ((32 - img.width) // 2, (32 - img.height) // 2))
        img = new_img
    else:
        img = img.resize((32, 32))

    # Display the image
    plt.imshow(img)
    plt.title("Resized Image")
    plt.axis("off")
    plt.show()
    
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img
