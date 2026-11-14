from src.models.wavenet import WaveNetClassifier
from src.train.torch_trainer import train


def build(in_channels, num_classes, args):
    return WaveNetClassifier(in_channels=in_channels, num_classes=num_classes, dropout=args.dropout)


if __name__ == "__main__":
    train("wavenet", build, {"epochs": 80, "lr": 1e-3})
