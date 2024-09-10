import os, shutil, argparse

MODELS_PATH = "./app/models"

def copy_model(models_path : str):
    '''
    A script that will create a python module package which contains the trained models.
    Args:
        - models_path : str , the path to the models
    Returns:
        - Null
    '''
    
    # first clean the packages model directory
    if os.path.exists(MODELS_PATH):
        files = os.listdir(MODELS_PATH)
        for file in files:
            if file == 'README.md':continue
            else: os.remove(os.path.join(MODELS_PATH, file))
    else:
        # create a models path
        os.mkdir(MODELS_PATH)

    # copy the models from the models directory to the ./src/release/ml_package/models
    models = os.listdir(path=models_path)
    for model in models:
        model_path = os.path.join(models_path , model)
        print(model_path)
        # copy the model to the new directory
        new_path = os.path.join(MODELS_PATH , model )
        shutil.copy(model_path , new_path)
    
    print('-- Model copied to package --')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--models_folder' , default='./models/model_object')
    args.add_argument('--test' , default='false')

    parsed_args = args.parse_args()

    models_folder = parsed_args.models_folder
    test_run = parsed_args.test

    copy_model(models_path=models_folder)