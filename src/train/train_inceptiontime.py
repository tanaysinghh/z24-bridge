from src.models.inceptiontime import InceptionTime
from src.train.torch_trainer import train


def build(in_channels, num_classes, args):
    return InceptionTime(in_channels=in_channels, num_classes=num_classes, dropout=args.dropout)


if __name__ == "__main__":
    train("inceptiontime", build, {"epochs": 80, "lr": 1e-3, "dropout": 0.0})
