import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

classes = [ 
            "Japanese Macaque",             #
            "Patas Monkey",                 #
            "Bald Uakari",                  #
            "Pygmy Marmoset",               
            "Silvery Marmoset",
            "White Headed Capuchin",
            "Nilgiri Langur",
            "Common Squirrel Monkey",
            "Black Headed Night Monkey", 
        ]

model = torch.load("best_model.pth")

mean = torch.tensor([0.4363, 0.4328, 0.3291])
std = torch.tensor([0.2137, 0.2083, 0.2046])

image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print (classes [predicted.item()])

classify(model, image_transforms, "monkey.jpg", classes)