import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

def load_image(image_path):
    """
    Load an image from the specified path
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    return Image.open(image_path).convert('RGB')

def resize_image(image, target_size):
    """
    Resize an image to the target size
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target size as (height, width)
        
    Returns:
        torch.Tensor: Resized image tensor
    """
    resize_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    
    return resize_transform(image)

def save_resized_image(tensor, output_path):
    """
    Save a tensor as an image
    
    Args:
        tensor (torch.Tensor): Image tensor
        output_path (str): Path to save the image
    """
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(output_path)
    
def main():
    parser = argparse.ArgumentParser(description='Resize an image using PyTorch')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output image')
    parser.add_argument('--height', type=int, required=True, help='Target height')
    parser.add_argument('--width', type=int, required=True, help='Target width')
    
    args = parser.parse_args()
    
    # Load image
    image = load_image(args.input)
    
    # Resize image
    target_size = (args.height, args.width)
    resized_tensor = resize_image(image, target_size)
    
    # Save the resized image
    save_resized_image(resized_tensor, args.output)
    print(f"Resized image saved to {args.output}")

if __name__ == "__main__":
    main()
