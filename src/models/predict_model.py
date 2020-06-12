from ..dirs import DIR_DATA_MODELS, DIR_DATA_RAW

if __name__ == '__main__':

    import torch
    import torchvision
    import numpy as np
    from .transform import get_X_scaled
    from ..features.make_features import make_feature_extraction

    model = torchvision.models.resnet50(pretrained=True,progress=True)

    first_conv_layer = model.conv1
    model.conv1= torch.nn.Sequential(
                            torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
                            first_conv_layer
    )  

    n_class = 12
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
                                torch.nn.Dropout(0.2),
                                torch.nn.Linear(num_ftrs, n_class)
    )


    checkpoint = torch.load(str(DIR_DATA_MODELS / 'best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    path_file = str(DIR_DATA_RAW / 'Alex/6.wav')
    inputs = make_feature_extraction(path_file) #Get MFCC params
    inputs = np.expand_dims(inputs, axis=0) # Add batch size = 1
    inputs = get_X_scaled(inputs) #Normalise
    inputs = torch.FloatTensor(inputs)

    predict_classes = { 0: 'female_angry',
                        1: 'female_disgust',
                        2: 'female_fear',
                        3: 'female_happy',
                        4: 'female_neutral',
                        5: 'female_sad',
                        6: 'male_angry',
                        7: 'male_disgust',
                        8: 'male_fear',
                        9: 'male_happy',
                        10: 'male_neutral',
                        11: 'male_sad'
                    }

    predict_model = torch.argmax(model(inputs))

    print('Alexandr 6 :', predict_classes[predict_model.item()])
    # print(torch.max(model(inputs), 1)[0].data)