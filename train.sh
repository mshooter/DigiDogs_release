NAME="TrainDigiDogs"
echo Logging at "logs/${NAME} and checkpoints are saved in checkpoints/${NAME}"
python scripts/launch.py fit --config digidogs/configs/train_digidogs.yaml \
--trainer.logger TensorBoardLogger \
--trainer.logger.name $NAME \
--trainer.logger.save_dir "logs/${NAME}" \
--trainer.callbacks+=ModelCheckpoint \
--trainer.callbacks.dirpath "checkpoints/${NAME}" \
--trainer.callbacks.save_last True \
--trainer.callbacks.save_top_k 3 \
--trainer.callbacks.filename {epoch}_{val_loss:.4f} \
--trainer.callbacks.monitor val_loss \
--trainer.callbacks.mode min \
--trainer.gradient_clip_val 0.5 \
--trainer.gradient_clip_algorithm value \
--trainer.detect_anomaly True
