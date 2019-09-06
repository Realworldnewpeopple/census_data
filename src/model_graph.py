from src.census import base_model, learning_curve_plot
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from src.census import base_model
import logging
from src.census import base_neural_model


class models:
    logging.basicConfig(filename="./log/info.log",
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode='w')
    logger = logging.getLogger(__name__)
    level = logging.getLevelName('INFO')
    logger.setLevel(level)

    def __init__(self, X_train, X_test, y_train,  y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def regression(self):
        model1 = linear_model.LogisticRegression(
            random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=20000)
        self.logger.info("Regression model starts training")
        regression = base_model(model1, self.X_train,
                                self.X_test, self.y_train, self.y_test)
        self.logger.info(
            "Prediction accuracy and list of cross_val_score is saved in score folder")
        cross1 = cross_val_score(model1, self.X_train, self.y_train, cv=10)
        self.logger.info("Plot is saved in image folder")
        learning_curve_plot(model1, self.X_train,
                            self.y_train, "Logistic Regression")
        return regression, cross1

    def decisiontree(self):
        model2 = DecisionTreeClassifier(criterion="entropy", max_depth=8)
        self.logger.info("decisiontree model starts training")
        deci = base_model(model2, self.X_train,
                          self.X_test, self.y_train, self.y_test)
        self.logger.info(
            "Prediction accuracy and list of cross_val_score is saved in score folder")
        cross2 = cross_val_score(model2, self.X_train, self.y_train, cv=10)
        self.logger.info("Plot is saved in image folder")
        learning_curve_plot(model2, self.X_train,
                            self.y_train, "Decision Tree")
        return deci, cross2

    def randomforrest(self):
        model3 = RandomForestClassifier(n_estimators=98, max_depth=10)
        self.logger.info("randomforrest model starts training")
        random = base_model(model3, self.X_train,
                            self.X_test, self.y_train, self.y_test)
        self.logger.info(
            "Prediction accuracy and list of cross_val_score is saved in score folder")
        cross3 = cross_val_score(model3, self.X_train, self.y_train, cv=10)
        self.logger.info("Plot is saved in image folder")
        learning_curve_plot(model3, self.X_train,
                            self.y_train, "random forrest")
        return random, cross3

    def supportvectorm(self):
        model4 = svm.SVC(gamma='scale', kernel='rbf',
                         decision_function_shape='ovr')
        self.logger.info("supportvectorm model starts training")
        support = base_model(model4, self.X_train,
                             self.X_test, self.y_train, self.y_test)
        self.logger.info(
            "Prediction accuracy and list of cross_val_score is saved in score folder")
        cross4 = cross_val_score(model4, self.X_train, self.y_train, cv=10)
        self.logger.info("Plot is saved in image folder")
        learning_curve_plot(model4, self.X_train,
                            self.y_train, "support vector machine")
        return support, cross4

    def KNeighborsClassifier_model(self):
        model5 = KNeighborsClassifier(n_neighbors=10)
        self.logger.info("KNeighborsClassifier model starts training")
        neighbor = base_model(model5, self.X_train,
                              self.X_test, self.y_train, self.y_test)
        self.logger.info(
            "Prediction accuracy and list of cross_val_score is saved in score folder")
        cross5 = cross_val_score(model5, self.X_train, self.y_train, cv=10)
        self.logger.info("Plot is saved in image folder")
        learning_curve_plot(model5, self.X_train,
                            self.y_train, "KNeighborsClassifier")
        return neighbor, cross5

    def train_neuralnet(self):
        base_neural_model(self.X_train, self.X_test, self.y_train, self.y_test)
