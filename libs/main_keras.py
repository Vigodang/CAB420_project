from libs import training_keras
from libs import models_keras
from libs import inference_keras
from libs import scoring
from libs import images2chips
from libs.config import USE_ELEVATION
import os
from tensorflow import keras

if __name__ == '__main__':
    dataset = 'data'

    image_chips = f'{dataset}/image-chips'
    label_chips = f'{dataset}/label-chips'
    eleva_chips = f'{dataset}/eleva-chips'
    chips_exist = os.path.exists(image_chips) and os.path.exists(label_chips)
    if USE_ELEVATION:
        chips_exist = chips_exist and os.path.exists(eleva_chips)

    if chips_exist:
        print('Chips already exist, skipping chip generation')
    else:
        images2chips.run(dataset)

    # train the model
    model = models_keras.build_unet(
        encoder='resnet18',
        pretrained=True,
        use_elevation=USE_ELEVATION,
    )
    
    # plot model architecture
    keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, dpi=100)
    print("Model architecture saved to model_architecture.png")
    
    training_keras.train_model(dataset, model, use_elevation=USE_ELEVATION)

    # use the train model to run inference on all test scenes
    inference_keras.run_inference(dataset, model=model, use_elevation=USE_ELEVATION)

    # scores all the test images compared to the ground truth labels then
    # send the scores (f1, precision, recall) and prediction images to wandb
    score, _ = scoring.score_predictions(dataset)
    print(score)
