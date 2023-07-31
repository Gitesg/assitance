  
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor

def load_pretrained_resnet():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_classes = 1000
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def classify_with_resnet(image_path, model):
    input_tensor = preprocess_image(image_path)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    return output

def main():
    image_path = 'path/to/your/image.jpg'
    model = load_pretrained_resnet()
    output = classify_with_resnet(image_path, model)

    print("Output tensor:")
    print(output)

if __name__ == "__main__":
    main()
