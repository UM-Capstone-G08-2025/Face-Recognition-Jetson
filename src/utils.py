def load_image(image_path):
    from PIL import Image
    import torchvision.transforms as transforms

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0)

def save_model(model, path):
    import torch
    torch.save(model.state_dict(), path)

def load_model(model, path):
    import torch
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        images.append(load_image(path))
    return images

def get_image_paths_from_directory(directory):
    import os
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]