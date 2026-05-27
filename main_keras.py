from libs import training_keras
from libs import models_keras
from libs import inference_keras
from libs import scoring
from libs import images2chips
import os

if __name__ == '__main__':
    dataset = 'data'

    image_chips = f'{dataset}/image-chips'
    label_chips = f'{dataset}/label-chips'
    eleva_chips = f'{dataset}/eleva-chips'
    if os.path.exists(image_chips) and os.path.exists(label_chips) and os.path.exists(eleva_chips):
        print('Chips already exist, skipping chip generation')
    else:
        images2chips.run(dataset)

    # train the model
    model = models_keras.build_unet(encoder='resnet18')
    training_keras.train_model(dataset, model)

    # use the train model to run inference on all test scenes
    inference_keras.run_inference(dataset, model=model)

    # scores all the test images compared to the ground truth labels then
    # send the scores (f1, precision, recall) and prediction images to wandb
    score, _ = scoring.score_predictions(dataset)
    print(score)
