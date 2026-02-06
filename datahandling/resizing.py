from PIL import Image, ImageOps
import os
from pathlib import Path
import random

def resize_image(image_path="dataset/15-Scene Image Dataset/15-Scene/00", target_path="dataset/resized/00", image_name="1.jpg"):
    
    origin_path = Path(image_path) / image_name
    target_path = Path(target_path) / image_name 

    if not origin_path.exists():
        print(f"The image {origin_path} doesn't exist")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"An image in {target_path} already exists")
    else:
        image = Image.open(origin_path)
        resized = image.resize((64, 64))
        resized = resized.convert("L")
        resized.save(target_path)
        print(f"Image saved to {target_path}")

    return



def mirror_image(image_path: str = "dataset/resized/00", target_path: str = "dataset/augmented/mirror", image_name: str = "1.jpg"):
    
    origin_path = Path(image_path) / image_name
    target_path = Path(target_path) / image_name 

    if not origin_path.exists():
        print(f"The image {origin_path} doesn't exist")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"An image in {target_path} already exists")
    else:
        mirror_image = Image.open(origin_path)
        mirror_image = ImageOps.mirror(mirror_image)
        mirror_image = mirror_image.convert("L")
        mirror_image.save(target_path)
        print(f"Image saved to {target_path}")

    return


def crop_image(image_path: str = "dataset/15-Scene Image Dataset/15-Scene/00", target_path: str = "dataset/augmented/cropping", image_name: str = "1.jpg"):
    
    origin_path = Path(image_path) / image_name
     

    if not origin_path.exists():
        print(f"The image {origin_path} doesn't exist")
        return

    augmentation_name_0 = f"crop_0_" + image_name
    augmented_path_0 = Path(target_path) / augmentation_name_0
    augmented_path_0.parent.mkdir(parents=True, exist_ok=True)

    # here i decided to do this control only on the frist name and not all 4 
    # because the operation are done to all 4 names, every time 
    if augmented_path_0.exists():
        print(f"An image in {augmented_path_0} already exists")

    else:
        augmentation_name_1 = f"crop_1_" + image_name
        augmented_path_1 = Path(target_path) / augmentation_name_1
        augmented_path_1.parent.mkdir(parents=True, exist_ok=True)

        augmentation_name_2 = f"crop_2_" + image_name
        augmented_path_2 = Path(target_path) / augmentation_name_2
        augmented_path_2.parent.mkdir(parents=True, exist_ok=True)

        augmentation_name_3 = f"crop_3_" + image_name
        augmented_path_3 = Path(target_path) / augmentation_name_3
        augmented_path_3.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(origin_path)
        w, h = image.size

        # I feel like this is not a proper way of cropping to have more images,  
        # in this case some parts of the room are always taken in consideration (the center)
        # others instead are rarely take in consideration
        top_x_0 = random.randint(0, w//8)
        top_y_0 = random.randint(0, h//8)
        bottom_x_0 = random.randint(top_x_0 + w//2, w - top_x_0)
        bottom_y_0 = random.randint(top_y_0 + h//2, h - top_y_0)

        bottom_x_1 = random.randint(w - w//8, w)
        bottom_y_1 = random.randint(h - h//8, h)
        top_x_1 = random.randint(w-bottom_x_1, bottom_x_1 - w//2)
        top_y_1 = random.randint(h-bottom_y_1, bottom_y_1 - h//2)

        cropped_image_0 = image.crop((top_x_0, top_y_0, bottom_x_0, bottom_y_0))
        cropped_image_0 = cropped_image_0.resize((64, 64))
        cropped_image_0 = cropped_image_0.convert("L")
        cropped_image_0.save(augmented_path_0)

        cropped_image_1 = image.crop((top_x_1, top_y_0, bottom_x_1, bottom_y_0))
        cropped_image_1 = cropped_image_1.resize((64, 64))
        cropped_image_1 = cropped_image_1.convert("L")
        cropped_image_1.save(augmented_path_1)

        cropped_image_2 = image.crop((top_x_0, top_y_1, bottom_x_0, bottom_y_1))
        cropped_image_2 = cropped_image_2.resize((64, 64))
        cropped_image_2.convert("L")
        cropped_image_2.save(augmented_path_2)

        cropped_image_3 = image.crop((top_x_1, top_y_1, bottom_x_1, bottom_y_1))
        cropped_image_3 = cropped_image_3.resize((64, 64))
        cropped_image_3 = cropped_image_3.convert("L")
        cropped_image_3.save(augmented_path_3)
        
        print(f"Image saved to {augmented_path_0}")

    return

## questo di seguito è il seguito per genereare le immagini modificate e ridimensionate,
## prima di usare fare test per vedere se funzionano bene i path:
#
base_directory_test = Path("./dataset/test")
base_directory_train = Path("./dataset/test")

# test 1

for dir in base_directory_test.iterdir():
    if not dir.is_dir():
        continue
    print(dir)
for dir in base_directory_train.iterdir():
    if not dir.is_dir():
        continue
    print(dir)

# test 2

for dir in base_directory_test.iterdir():
    if not dir.is_dir():
        continue
    for img in dir.iterdir():
        print(img.name)

for dir in base_directory_train.iterdir():
    if not dir.is_dir():
        continue
    for img in dir.iterdir():
        print(img.name)


for dir in base_directory_test.iterdir():
    if not dir.is_dir():
        continue
    target_resize_path = "dataset/resized/test/" + dir.name
    for img in dir.iterdir():
        resize_image(dir, target_resize_path, img.name)

for dir in base_directory_train.iterdir():
    if not dir.is_dir():
        continue
    target_resize_path = "dataset/resized/train/" + dir.name
    target_cropping_path = "dataset/augmented/cropping/" + dir.name
    for img in dir.iterdir():
        resize_image(dir, target_resize_path, img.name)
        crop_image(dir, target_cropping_path, img.name)

resize_path = Path("dataset/resized/train") 
for dir in resize_path.iterdir():
    if not dir.is_dir():
        continue
    target_mirror_path = "dataset/augmented/mirror/" + dir.name
    for img in dir.iterdir():
        mirror_image(dir, target_mirror_path, img.name)


## check grayscale and convert if not 
path_to_images = Path("dataset/resized/test")
for dir in path_to_images.iterdir():
    if not dir.is_dir():
        continue
    for path_img in dir.iterdir():
        img = Image.open(path_img)
        if img.mode != "L":
            print(f"Image {path_img.name} is NOT grayscale")
            img = img.convert("L")
            img.save(path_img)

path_to_images = Path("dataset/resized/train")
for dir in path_to_images.iterdir():
    if not dir.is_dir():
        continue
    for path_img in dir.iterdir():
        img = Image.open(path_img)
        if img.mode != "L":
            print(f"Image {path_img.name} is NOT grayscale")
            img = img.convert("L")
            img.save(path_img)

path_to_images_2 = Path("dataset/augmented/mirror")
for dir in path_to_images_2.iterdir():
    if not dir.is_dir():
        continue
    for path_img in dir.iterdir():
        img = Image.open(path_img)
        if img.mode != "L":
            print(f"Image {path_img.name} is NOT grayscale")
            img = img.convert("L")
            img.save(path_img)

path_to_images_3 = Path("dataset/augmented/cropping")
for dir in path_to_images_3.iterdir():
    if not dir.is_dir():
        continue
    for path_img in dir.iterdir():
        img = Image.open(path_img)
        if img.mode != "L":
            print(f"Image {path_img.name} is NOT grayscale")
            img = img.convert("L")
            img.save(path_img)