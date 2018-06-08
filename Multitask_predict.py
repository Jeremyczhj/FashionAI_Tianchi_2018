import gc
import pandas as pd
from keras.layers import *
from keras.models import *
import inception_v4
from keras.preprocessing.image import *
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications import *
from dataset import *
from config import *

def predict(task):

    if(task=='design'):
        task_list = task_list_design
        model1_path = MODEL_DESIGN_INCEPTIONV4
        model2_path = MODEL_DESIGN_INCEPTIONRESNETV2
    else:
        task_list = task_list_length
        model1_path = MODEL_LENGTH_INCEPTIONV4
        model2_path = MODEL_LENGTH_INCEPTIONRESNETV2
    label_names = list(task_list.keys())
    
    # load model 1
    base_model = inception_v4.create_model(weights='imagenet', include_top=False, width=width)
    input_tensor = Input((width, width, 3))
    x = input_tensor
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in task_list.items()]

    model1 = Model(input_tensor, x)
    model1.load_weights(model1_path, by_name=True)

    y_pred11 = process('default', model1, width, fnames_test, n_test)
    y_pred12 = process('flip', model1, width, fnames_test, n_test)

    del model1,base_model,x,input_tensor

    # load model 2
    base_model2 = InceptionResNetV2(weights='imagenet',input_shape=(width, width, 3),include_top=False)
    input_tensor2 = Input((width, width, 3))
    x2 = input_tensor2
    x2 = Lambda(preprocess_input, name='preprocessing')(x2)
    x2 = base_model2(x2)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dropout(0.5)(x2)
    x2 = [Dense(count, activation='softmax', name=name)(x2) for name, count in task_list.items()]

    model2 = Model(input_tensor2, x2)
    model2.load_weights(model2_path, by_name=True)

    y_pred21 = process('default', model2, width, fnames_test, n_test)
    y_pred22 = process('flip', model2, width, fnames_test, n_test)
    
    # ensemble two models
    for i in range(n_test):
        problem_name = df_test.label_name[i].replace('_labels', '')
        problem_index = label_names.index(problem_name)
        probs11 = y_pred11[problem_index][i]
        probs12 = y_pred12[problem_index][i]
        probs21 = y_pred21[problem_index][i]
        probs22 = y_pred22[problem_index][i]

        probs1 = probs11 + probs12
        probs2 = probs21 + probs22

        probs1 = probs1 / 2
        probs2 = probs2 / 2

        probs = 0.5*probs1+0.5*probs2

        df_test.label[i] = ';'.join(np.char.mod('%.8f', probs))
    
    # write csv files
    fname_csv = 'result/%s.csv' % (task)
    df_test.to_csv(fname_csv, index=None, header=None)

    del model2


def csv_loader():
    df_test = pd.read_csv(TEST_LABEL_DIR, header=None)
    df_test.columns = ['filename', 'label_name', 'label']

    df_test_length = df_test[(df_test.label_name == 'skirt_length_labels') | (df_test.label_name == 'sleeve_length_labels')
                          |(df_test.label_name == 'coat_length_labels')|(df_test.label_name == 'pant_length_labels')]

    df_test_design = df_test[(df_test.label_name == 'collar_design_labels') | (df_test.label_name == 'lapel_design_labels')
                          | (df_test.label_name == 'neckline_design_labels') | (df_test.label_name == 'neck_design_labels')]
    df_test_length.to_csv(TEST_LENGTH_LABEL_DIR, index=False, header=None)
    df_test_design.to_csv(TEST_DESIGN_LABEL_DIR, index=False, header=None)


if __name__ == "__main__":
    csv_loader()

    df_test = pd.read_csv(TEST_DESIGN_LABEL_DIR, header=None)
    df_test.columns = ['filename', 'label_name', 'label']
    fnames_test = df_test.filename
    n_test = len(df_test)
    predict('design')
    del df_test

    df_test = pd.read_csv(TEST_LENGTH_LABEL_DIR, header=None)
    df_test.columns = ['filename', 'label_name', 'label']
    fnames_test = df_test.filename
    n_test = len(df_test)
    predict('length')
    del df_test

    gc.collect()

