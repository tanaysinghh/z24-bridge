from src.models.inceptiontime import InceptionTime
from src.train.torch_trainer import train

if __name__ == "__main__":
    train("inceptiontime", lambda: InceptionTime(in_channels=1, num_classes=17), {"epochs": 80, "lr": 1e-3})
