import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.image import flip_left_right, flip_up_down, adjust_brightness
import matplotlib.pyplot as plt
import os


def sort_by_name(arr, split_at):
    return sorted([f for f in arr if f.split(split_at)[0].isdigit()], key=lambda x: int(x.split(split_at)[0]))


def load_data(dir_path):
    # dir_path = '../satellite-roads/train/'
    directory = os.listdir(dir_path)
    images = []
    masks = []

    for filename in directory:
        if filename.split('.')[1] == 'jpg':
            images.append(filename)
        elif filename.split('.')[1] == 'png':
            masks.append(filename)

    #sorted_images = sort_by_name(images, '_')
    #sorted_masks = sort_by_name(masks, '_')

    return np.array(images), np.array(masks)

'''
def preprocess_data(root_path, images, masks, input_size, augmented=False):

    image = []
    maske = []
    for img_file, mask_file in zip(images, masks):
        img = load_img(root_path + img_file, target_size=input_size, color_mode='rgb')
        mask = load_img(root_path + mask_file, target_size=input_size, color_mode='grayscale')

        # Convert image and mask to arrays
        img_array = img_to_array(img)
        img_array = img_array / 255.0

        mask_array = img_to_array(mask, dtype=np.bool_)

        # Append images and masks to the lists
        image.append(img_array)
        maske.append(mask_array)

        if augmented:
            images.append(flip_left_right(img_array))
            maske.append(flip_left_right(mask_array))

            images.append(flip_up_down(img_array))
            maske.append(flip_up_down(mask_array))

    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(maske)

    return images, masks



def preprocess_data(root_path, images, masks, input_size, augmented=False):
    def flip_left_right(image, mask):
        return tf.image.flip_left_right(image), tf.image.flip_left_right(mask)

    def flip_up_down(image, mask):
        return tf.image.flip_up_down(image), tf.image.flip_up_down(mask)

    def rotate_image(image, mask):
        k = np.random.randint(1, 4)  # Randomly choose 90, 180, or 270 degrees
        return tf.image.rot90(image, k=k), tf.image.rot90(mask, k=k)

    def adjust_brightness(image, mask):
        image = tf.image.random_brightness(image, max_delta=0.2)  # Brightness delta range
        return image, mask

    def random_zoom(image, mask, zoom_range=(0.8, 1.2)):
        scale = np.random.uniform(zoom_range[0], zoom_range[1])
        img_resized = tf.image.resize(image, [int(image.shape[0] * scale), int(image.shape[1] * scale)])
        mask_resized = tf.image.resize(mask, [int(mask.shape[0] * scale), int(mask.shape[1] * scale)])
        img_cropped = tf.image.resize_with_crop_or_pad(img_resized, image.shape[0], image.shape[1])
        mask_cropped = tf.image.resize_with_crop_or_pad(mask_resized, mask.shape[0], mask.shape[1])
        return img_cropped, mask_cropped

    image_list = []
    mask_list = []

    for img_file, mask_file in zip(images, masks):
        img = load_img(root_path + img_file, target_size=input_size, color_mode='rgb')
        mask = load_img(root_path + mask_file, target_size=input_size, color_mode='grayscale')

        # Convert image and mask to arrays
        img_array = img_to_array(img) / 255.0  # Normalize image
        mask_array = img_to_array(mask, dtype=np.bool_)  # Binary mask

        # Add original image and mask
        image_list.append(img_array)
        mask_list.append(mask_array)

        if augmented:
            # Apply augmentations
            img_aug, mask_aug = flip_left_right(img_array, mask_array)
            image_list.append(img_aug)
            mask_list.append(mask_aug)

            img_aug, mask_aug = flip_up_down(img_array, mask_array)
            image_list.append(img_aug)
            mask_list.append(mask_aug)

            img_aug, mask_aug = rotate_image(img_array, mask_array)
            image_list.append(img_aug)
            mask_list.append(mask_aug)

            img_aug, mask_aug = adjust_brightness(img_array, mask_array)
            image_list.append(img_aug)
            mask_list.append(mask_aug)

            img_aug, mask_aug = random_zoom(img_array, mask_array)
            image_list.append(img_aug)
            mask_list.append(mask_aug)

    # Convert lists to numpy arrays
    images_array = np.array(image_list)
    masks_array = np.array(mask_list)

    return images_array, masks_array
'''
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.composition import OneOf


def preprocess_data(root_path, images, masks, input_size, augmented=False):
    # Define Albumentations augmentation pipeline
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),  # Random rotation within -30 to +30 degrees
        A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
        A.GaussianBlur(p=0.2),  # Apply Gaussian blur
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Elastic distortion
    ])
    
    image_list = []
    mask_list = []
    
    for img_file, mask_file in zip(images, masks):
        # Load image and mask
        img = load_img(root_path + img_file, target_size=input_size, color_mode='rgb')
        mask = load_img(root_path + mask_file, target_size=input_size, color_mode='grayscale')
        
        # Convert image and mask to arrays
        img_array = img_to_array(img) / 255.0  # Normalize image
        mask_array = img_to_array(mask, dtype=np.bool_)  # Binary mask
        
        # Add original image and mask
        image_list.append(img_array)
        mask_list.append(mask_array)
        
        if augmented:
            # Apply augmentations
            augmented_data = augmentation_pipeline(image=img_array, mask=mask_array)
            img_aug = augmented_data['image']
            mask_aug = augmented_data['mask']
            
            # Add augmented image and mask
            image_list.append(img_aug)
            mask_list.append(mask_aug)
    
    # Convert lists to numpy arrays
    images_array = np.array(image_list)
    masks_array = np.array(mask_list)
    
    return images_array, masks_array

def display_data(dir_path, image_paths, mask_paths):

    fig, axes = plt.subplots(5, 2, figsize=(10, 20))

    # Iterate over the image and mask pairs and display them in subplots
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load the image and mask using your preferred method
        image = plt.imread(dir_path + image_path)
        mask = plt.imread(dir_path + mask_path)

        # Plot the image and mask in the corresponding subplot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

