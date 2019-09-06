import json
from src.model_graph import models
from src.jsonstore import json_update
from src import census
#from keras.models import load_model


def main(config):
    data = census.cat_num(census.get_data(config['path']))
    data=census.scaling_data(data)
    census.freq_dist(data)
    X_train, X_test, y_train, y_test = census.train_test_split_data(
        data)
    model_gr = models(X_train, X_test, y_train, y_test)
    if config['LogisticRegression'] == "True":
        regression, cross1 = model_gr.regression()
        json_update(regression, cross1, "logistic_regresstion")
    if config['DecisionTreeClassifier'] == "True":
        Tree, cross2 = model_gr.decisiontree()
        json_update(Tree, cross2, "DecisionTreeClassifier")
    if config['RandomForestClassifier'] == "True":
        forest, cross3 = model_gr.randomforrest()
        json_update(forest, cross3, "RandomForestClassifier")
    if config['SupportVectorMachine'] == "True":
        sup, cross4 = model_gr.supportvectorm()
        json_update(sup, cross4, "SupportVectorMachine")
    if config['KNeighborsClassifier'] == "True":
        neigh, cross5 = model_gr.KNeighborsClassifier_model()
        json_update(neigh, cross5, "KNeighborsClassifier")
    if config['traintestnet'] == "True":
        model_gr.train_neuralnet()


if __name__ == "__main__":
    with open('./config/config.json') as json_file:
        config = json.load(json_file)
    main(config)
