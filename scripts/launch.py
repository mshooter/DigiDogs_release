#data_dir: "/vol/research/datasets/mixedmode/gtadogs" 
from lightning.pytorch.cli import LightningCLI
from digidogs.datasets.datamoduleposev2 import GtaModule
from digidogs.models.litposev2 import LitPoser

if __name__ == "__main__": 
    cli = LightningCLI(LitPoser, GtaModule)

