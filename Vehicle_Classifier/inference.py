import argparse
from PIL import Image

import torch
import torchvision

import tvm
from tvm import relay

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

color_attrs = ['Black', 'Blue', 'Brown','Gray', 'Green', 'Pink', 'Red', 'White', 'Yellow']
direction_attrs = ['Front', 'Rear']
type_attrs = ['passengerCar', 'saloonCar', 'shopTruck', 'suv', 'trailer', 'truck', 'van', 'waggon']

class Cls_Net(torch.nn.Module):
    """
    vehicle multilabel classification model
    """

    def __init__(self, num_cls, input_size):
        """
        network definition
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        # delete original FC and add custom FC
        self.features = torchvision.models.resnet18(pretrained=True)
        del self.features.fc

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(512 ** 2, num_cls)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        X = self.features(X)  # extract features

        X = X.view(N, 512, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN

        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self._num_cls)
        return X

def run(model, image_path, backend="torch"):
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    image = Image.open(image_path)
    image = image.convert('RGB')
    image = transforms(image)
    image = image.view(1, 3, 224, 224)
    
    output = None

    # infer at fp16
    model.half().to("cuda:0")
    image = image.half().to("cuda:0")

    output = model.forward(image).cpu()

    pred_color = output[:, :9].max(1, keepdim=True)[1]
    pred_direction = output[:, 9:11].max(1, keepdim=True)[1]
    pred_type = output[:, 11:].max(1, keepdim=True)[1]

    print(f"test image: {image_path}, ",
        f"color is: {color_attrs[pred_color[0][0]]}, ",
        f"direction is: {direction_attrs[pred_direction[0][0]]}, ", 
        f"type is: {type_attrs[pred_type[0][0]]}"
    )

def main():
    parser = argparse.ArgumentParser(description="Test Vehicle Classifier")
    parser.add_argument("-w", "--weights", default=1, type=str)
    parser.add_argument("-e", "--export", type=str)
    parser.add_argument('-i', "--image", type=str)
    args = parser.parse_args()

    model = None

    model = Cls_Net(num_cls=19, input_size=224)

    model.load_state_dict(torch.load(args.weights))

    model.eval()

    if args.export:
        input_names = ['input']

        output_names = ['output']

        dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}

        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model, 
            dummy_input, 
            args.export, 
            input_names = input_names, 
            dynamic_axes = dynamic_axes, 
            output_names = output_names,
            opset_version=13
        )
        print("Export Onnx Success!")
        return True
        
    run(model, args.image)    

if __name__ == "__main__":
    main()
    