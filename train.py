import argparse
import torch
from linking.config.config import Config
from linking.data.dataloader import create_data
from linking.model.build_model import build_model
from linking.training.trainer import Trainer
import os

from linking.training.trainer_qed import QEDTrainer

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="/Users/padr/repos/linking/configs/default_config.json",
        help="Path to project config file",
        required=False,
    )
    args = parser.parse_args()

    # parse json to dict
    # with open(args.config) as json_config:
    #    config = json.load(json_config)

    # Create config object
    config = Config()
    # config = Config(**config) # Use this if passing a config json instead of defaults

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data
    data = create_data(config=config, device=device)

    # Create Model
    model = build_model(config=config, device=device)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create Trainer
    if config.model == "TeacherForcer":
        trainer = Trainer(model=model, data=data, optimizer=optimizer, config=config)
    elif config.model == "QED":
        trainer = QEDTrainer(model=model, data=data, optimizer=optimizer, config=config)

    # Train
    trainer.train()
    trainer.test()

    if not os.path.exists(config.save_model):
        os.makedirs(config.save_model)

    torch.save(model, config.save_model + 'model')