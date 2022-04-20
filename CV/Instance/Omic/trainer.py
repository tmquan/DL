import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from data import PseudoDataModule 
from model import PixelEmbeddingUNetModule


if __name__ == '__main__':
    batch_size=20
    lr=1e-4
    max_epochs=2001
    
    datamodule = PseudoDataModule(
        batch_size=batch_size
    )
    model = PixelEmbeddingUNetModule()
    log_dir = "/data/qtran/logs_omics"
    tsb_logger = TensorBoardLogger(save_dir=os.path.join(log_dir, 'tsb'))

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=log_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = Trainer(
        accelerator="gpu", 
        devices=[1],
        max_epochs=max_epochs,
        logger=[tsb_logger],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    trainer.fit(model, datamodule)

    checkpoint_callback.best_model_path
    trainer.test(datamodule=datamodule, 
                 ckpt_path=checkpoint_callback.best_model_path)