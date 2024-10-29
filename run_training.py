if __name__ == '__main__':
    from pet_seg_core.config import PetSegTrainConfig
    PetSegTrainConfig.EPOCHS = 5
    PetSegTrainConfig.TOTAL_SAMPLES = -1
    PetSegTrainConfig.DESCRIPTION_TEXT = "UNET with RGB input and 3 channel(3 class) output, trained on all samples for 5 epochs. Will be used for webapp"
    from pet_seg_core.train import train
    train()