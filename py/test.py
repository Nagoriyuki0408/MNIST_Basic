import torch
from PIL import Image
from torchvision import transforms
from py.model import CNN

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # 添加batch维度

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load('../mnist_cnn.pth'))
    model.eval()

    with torch.no_grad():
        output = model(img.to(device))
        _, predicted = torch.max(output, 1)
        print(f'预测结果：{predicted.item()}')

predict_image('/9.png') # 替换为你的图片路径

