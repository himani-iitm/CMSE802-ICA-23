import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

# ----- Helper Functions -----
def load_image_from_url(url, device, image_size=224):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = preprocess(img).unsqueeze(0).to(device)
    return img

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = 0

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                style_img, content_img,
                                content_layers, style_layers):

    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    return optim.LBFGS([input_img.requires_grad_()])

# ----- Main Function -----
def main():
    parser = argparse.ArgumentParser(description='Neural Style Transfer Script')
    parser.add_argument('--content_url', type=str, required=True)
    parser.add_argument('--style_url', type=str, required=True)
    parser.add_argument('--content_layers', nargs='+', default=['conv_2'])
    parser.add_argument('--style_layers', nargs='+', default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'])
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--style_weight', type=float, default=1000000)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--output_path', type=str, default='stylized_output.png')

    args = parser.parse_args()

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std  = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_img = load_image_from_url(args.content_url, device)
    style_img = load_image_from_url(args.style_url, device)
    input_img = content_img.clone()

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std,
        style_img, content_img,
        args.content_layers, args.style_layers)

    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= args.num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses) * args.style_weight
            content_score = sum(cl.loss for cl in content_losses) * args.content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    output_img = input_img.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_img)
    output_img.save(args.output_path)

if __name__ == '__main__':
    main()

