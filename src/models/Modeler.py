from __future__ import annotations

from typing import List, Optional, Union, NamedTuple, Dict

import pandas as pd
import numpy as np

import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, recall_score, precision_score, roc_auc_score, accuracy_score, \
    r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVR, SVC


class ClassifierMetrics(NamedTuple):
    class_report: Dict[str, float]
    accuracy: float
    recall: float
    precision: float
    roc_auc: float


class RegressionMetrics(NamedTuple):
    r2: float
    mean_absolute_error: float
    mean_squared_error: float


class Modeler:

    def __init__(
            self,
            df: pd.DataFrame,
            target_column: str,
    ) -> None:

        self.classifier_metrics: ClassifierMetrics = ClassifierMetrics(np.nan, np.nan, np.nan, np.nan, np.nan)
        self.regression_metrics: RegressionMetrics = RegressionMetrics(np.nan, np.nan, np.nan)
        self.classifier_estimator: Optional[
            Union[LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVR, SVC]] = None
        self.regression_estimator: Optional[
            Union[LinearRegression, RandomForestRegressor, GradientBoostingRegressor, SVR]] = None
        self.y_test: Optional[pd.Series] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.y_validation: Optional[pd.Series] = None
        self.X_validation: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.target_ps: Optional[pd.Series] = None
        self.scaled_df: Optional[pd.DataFrame] = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.one_hot_encoded_df: Optional[pd.DataFrame] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.numerical_imputer: Optional[SimpleImputer] = None
        self.original_df: pd.DataFrame = df

        self.df: pd.DataFrame = df
        self.target_column: str = target_column

    def impute_numerical(
            self,
            strategy: str
    ) -> Modeler:
        """
        Parameters
        ----------
        strategy : str
            The imputation strategy to use. Must be one of the following:
                - "mean": Impute missing values with the mean of the column.
                - "median": Impute missing values with the median of the column.
                - "most_frequent": Impute missing values with the most frequent value of the column.

        Returns
        -------
        Modeler
            The updated instance of the Modeler class with missing numerical values imputed.
        """
        valid_strategy: set = {"mean", "median", "most_frequent"}

        if strategy not in valid_strategy:
            raise ValueError(f"Strategy must be one of {valid_strategy}")

        self.numerical_imputer: SimpleImputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

        categorical_df: pd.DataFrame = self.df.select_dtypes(include=["object", "datetime64"]).copy()
        numerical_df: pd.DataFrame = self.df.select_dtypes(include=[np.number]).copy()

        self.numerical_imputer.fit(numerical_df)

        transformed_np: np.ndarray = self.numerical_imputer.transform(numerical_df)

        transformed_df: pd.DataFrame = pd.DataFrame(transformed_np, columns=numerical_df.columns)

        temp_df: pd.DataFrame = pd.concat([categorical_df, transformed_df], axis=1)

        self.df = temp_df

        return self

    def impute_categorical(
            self,
            strategy: str = "most_frequent",
            fill_value: Optional[str] = None
    ) -> Modeler:
        """
        Parameters
        ----------
        strategy : str
            The imputation strategy to use. Valid options are "most_frequent" and "constant".

        fill_value : Optional[str], default None
            The value to fill in missing values when the strategy is set to "constant". This parameter is optional
            and must be specified when the strategy is set to "constant".

        Returns
        -------
        Modeler
            Returns the instance of the Modeler class to allow method chaining.
        """

        valid_strategy: set = {"most_frequent", "constant"}

        if strategy not in valid_strategy:
            raise ValueError(f"Strategy must be one of {valid_strategy}")

        if (strategy == "constant") and fill_value is None:
            raise ValueError(f"fill_value must be set when strategy is set to `constant`")

        self.categorical_imputer: SimpleImputer = SimpleImputer(
            missing_values=np.nan,
            strategy=strategy,
            fill_value=fill_value
        )

        categorical_df: pd.DataFrame = self.df.select_dtypes(include=["object"]).copy()
        numerical_df: pd.DataFrame = self.df.select_dtypes(include=[np.number]).copy()

        self.categorical_imputer.fit(categorical_df)

        transformed_np: np.ndarray = self.categorical_imputer.transform(categorical_df)

        transformed_df: pd.DataFrame = pd.DataFrame(transformed_np, columns=categorical_df.columns)

        temp_df: pd.DataFrame = pd.concat([transformed_df, numerical_df], axis=1)

        self.df = temp_df

        return self

    def convert_categorical_to_one_hot(
            self
    ) -> Modeler:
        """
        Converts categorical variables in a DataFrame to one-hot encoded format.

        Parameters:
        - self: An instance of the Modeler class.

        Returns:
        - Modeler: An instance of the Modeler class with the one-hot encoded DataFrame stored in the `one_hot_encoded_df` attribute.
        """

        categorical_df: pd.DataFrame = self.df.select_dtypes(include=["object"]).copy()
        numerical_df: pd.DataFrame = self.df.select_dtypes(include=[np.number]).copy()

        one_hot_df: pd.DataFrame = pd.get_dummies(
            data=categorical_df,
        )

        merged_df: pd.DataFrame = pd.concat([one_hot_df, numerical_df], axis=1)

        self.one_hot_encoded_df: pd.DataFrame = merged_df

        return self

    def scale(
            self,
            strategy: str = "standard"
    ) -> Modeler:
        """
        Parameters
        ----------
        strategy : str, optional
            The scaling strategy to use. Default is "standard". Must be one of {"standard", "minmax"}.

        Returns
        -------
        Modeler
            An instance of the Modeler class after applying the scaling operation.

        Raises
        ------
        ValueError
            If the provided strategy is not one of {"standard", "minmax"}.
        """

        valid_strategy: set = {"standard", "minmax"}

        if strategy not in valid_strategy:
            raise ValueError(f"Strategy must be one of {valid_strategy}")

        if strategy == "standard":
            self.scaler = StandardScaler()
        elif strategy == "minmax":
            self.scaler = MinMaxScaler()

        category_df: pd.DataFrame = self.df.select_dtypes(include=["object"]).copy()
        numerical_df: pd.DataFrame = self.df.select_dtypes(include=[np.number]).copy()

        scaled_np: np.ndarray = self.scaler.fit_transform(numerical_df)
        scaled_df: pd.DataFrame = pd.DataFrame(scaled_np, columns=numerical_df.columns)

        merged_df: pd.DataFrame = pd.concat(
            [
                category_df,
                scaled_df
            ], axis=1
        )

        if self.scaled_df is not None:
            self.scaled_df.reset_index(drop=True, inplace=True)
            merged_df.reset_index(drop=True, inplace=True)

        self.scaled_df: pd.DataFrame = merged_df
        self.df = merged_df

        if self.one_hot_encoded_df is not None:

            self.one_hot_encoded_df.reset_index(drop=True, inplace=True)
            merged_df.reset_index(drop=True, inplace=True)

            one_hot_columns: List[str] = self.one_hot_encoded_df.columns.to_list()
            scaled_columns: List[str] = scaled_df.columns.to_list()

            intersection_columns: List[str] = [each_col for each_col in one_hot_columns if each_col in scaled_columns]

            for each_col in intersection_columns:
                self.one_hot_encoded_df[each_col] = scaled_df[each_col]

        return self

    def train_test_split(
            self,
            test_proportion: float = 0.2,
            validation_proportion: float = 0.0
    ) -> Modeler:
        """
        Parameters
        ----------
        test_proportion : float, optional
            Proportion of the data to be used for testing. Default is 0.2.

        validation_proportion : float, optional
            Proportion of the data to be used for validation. Default is 0.0.

        Returns
        -------
        Modeler
            Returns the updated instance of the Modeler class.

        Raises
        ------
        ValueError
            If any of the proportions (test_proportion, validation_proportion) is greater than 1.
            If test_proportion or validation_proportion is larger than 0.5.

        Notes
        -----
        This method splits the data into training, testing, and validation sets.
        The target variable is separated from the features.

        If the data has categorical columns, it is important to either convert them to numerical values
        using one-hot encoding or drop them from the dataset.

        Before the split, any rows containing null values are removed from the dataset.

        If the validation_proportion argument is specified, a validation set is also created.

        Examples
        --------
        >>> modeler = Modeler()
        >>> modeler.train_test_split(test_proportion=0.2, validation_proportion=0.1)
        """

        train_proportion: float = 1 - test_proportion

        if any(proportion > 1 for proportion in [train_proportion, test_proportion, validation_proportion]):
            raise ValueError("Proportions must not be more than 1")

        if train_proportion < max(test_proportion, validation_proportion):
            raise ValueError("test_proportion and/or validation_proportion cannot > 0.5")

        # remove any remaining `null` before train_test split
        self.df = self.df.dropna(axis=0, how="any")

        if self.one_hot_encoded_df is not None:
            self.one_hot_encoded_df = self.one_hot_encoded_df.dropna(axis=0, how="any")

        non_numeric_columns: List[str] = self.df.select_dtypes(exclude=[np.number]).columns.to_list()

        if len(non_numeric_columns) > 0:

            if self.one_hot_encoded_df is None:
                raise ValueError("You have categorical columns in the features. You can either convert them"
                                 "into numerical (e.g., using one-hot encoding)"
                                 ", or drop these categorical variables")

            self.target_ps: pd.Series = self.one_hot_encoded_df[self.target_column].copy()
            self.feature_df: pd.DataFrame = self.one_hot_encoded_df.drop(labels=self.target_column, axis=1)

        else:

            self.target_ps: pd.Series = self.df[self.target_column].copy()
            self.feature_df: pd.DataFrame = self.df.drop(labels=self.target_column, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_df,
            self.target_ps,
            test_size=test_proportion,
            random_state=42
        )

        if validation_proportion > 0:
            X_train, X_validation, y_train, y_validation = train_test_split(
                X_train,
                y_train,
                test_size=validation_proportion,
                random_state=45
            )

            self.X_validation: pd.DataFrame = X_validation
            self.y_validation: pd.Series = y_validation

        self.X_train: pd.DataFrame = X_train
        self.X_test: pd.DataFrame = X_test

        self.y_train: pd.Series = y_train
        self.y_test: pd.Series = y_test

        return self

    def train_regression_estimator(
            self,
            which_regressor: str = "random_forest"
    ) -> Modeler:
        """
        Parameters
        ----------
        which_regressor : str
            The type of regression estimator to use. Possible values are "linear_regression",
            "random_forest", "gradient_boost", and "support_vector".

        Returns
        -------
        Modeler
            The trained regression estimator.

        Raises
        ------
        ValueError
            If the given `which_regressor` is not a valid option.

        Notes
        -----
        This method trains a regression estimator using the specified algorithm. The algorithm
        is chosen based on the value of the `which_regressor` parameter. The trained estimator
        is then stored in `self.regression_estimator`. The training data is provided as `X_train`
        and `y_train`, which are assumed to be previously set in the class.

        Example
        -------
        >>> estimator = train_regression_estimator(which_regressor="random_forest")
        """

        valid_regressor: set = {
            "linear_regression",
            "random_forest",
            "gradient_boost",
            "support_vector"
        }

        if which_regressor not in valid_regressor:
            raise ValueError(f"which_regressor {which_regressor} not in {valid_regressor}")

        # Select appropriate regression estimator
        if which_regressor == "linear_regression":
            self.regression_estimator: LinearRegression = LinearRegression()

        elif which_regressor == "random_forest":
            self.regression_estimator: RandomForestRegressor = RandomForestRegressor()

        elif which_regressor == "gradient_boost":
            self.regression_estimator: GradientBoostingRegressor = GradientBoostingRegressor()

        elif which_regressor == "support_vector":
            self.regression_estimator: SVR = SVR()

        # train the regression_estimator
        self.regression_estimator.fit(
            X=self.X_train,
            y=self.y_train,
        )

        return self

    def train_classifier_estimator(
            self,
            which_classifier: str = "random_forest"
    ) -> Modeler:
        """
        Parameters
        ----------
        which_classifier: str
            The type of classifier to be used for training. The value must be one of the following:
            - "logistic_regression" for training with Logistic Regression
            - "random_forest" for training with Random Forest
            - "gradient_boost" for training with Gradient Boosting
            - "support_vector" for training with Support Vector Machine

        Returns
        -------
        Modeler
            The trained classifier estimator.

        Raises
        ------
        ValueError
            If the `which_classifier` parameter is not one of the valid classifier types.

        Notes
        -----
        This method trains a classifier estimator using the specified classifier type. The train data is taken
        from `self.X_train` and `self.y_train`. The trained classifier estimator is then returned.
        """

        valid_classifier: set = {
            "logistic_regression",
            "random_forest",
            "gradient_boost",
            "support_vector"
        }

        if which_classifier not in valid_classifier:
            raise ValueError(f"which_classifier {which_classifier} not in {valid_classifier}")

        if which_classifier == "logistic_regression":
            self.classifier_estimator: LogisticRegression = LogisticRegression()

        if which_classifier == "random_forest":
            self.classifier_estimator: RandomForestClassifier = RandomForestClassifier()

        if which_classifier == "gradient_boost":
            self.classifier_estimator: GradientBoostingClassifier = GradientBoostingClassifier()

        if which_classifier == "support_vector":
            self.classifier_estimator: SVR = SVR()

        self.classifier_estimator.fit(
            X=self.X_train,
            Y=self.y_train,
        )

        return self

    def calculate_classifier_metrics(
            self
    ) -> Modeler:

        """

        This method calculates and stores various performance metrics for a classifier model.

        Parameters:
            self: The object instance.

        Returns:
            Modeler: The object instance with updated classifier metrics.

        Example usage:
            model = Modeler()
            model.calculate_classifier_metrics()

        """

        y_pred: pd.Series = self.classifier_estimator.predict(self.X_test)

        class_report = classification_report(y_true=self.y_test, y_pred=y_pred)
        accuracy = accuracy_score(y_true=self.y_test, y_pred=y_pred)
        recall = recall_score(y_true=self.y_test, y_pred=y_pred)
        precision = precision_score(y_true=self.y_test, y_pred=y_pred)
        roc_auc = roc_auc_score(y_true=self.y_test, y_score=y_pred)

        classifier_metrics: ClassifierMetrics = ClassifierMetrics(
            class_report=class_report,
            accuracy=accuracy,
            recall=recall,
            precision=precision,
            roc_auc=roc_auc,
        )

        self.classifier_metrics = classifier_metrics

        return self

    def calculate_regression_metrics(
            self
    ) -> Modeler:

        """
        Calculate regression metrics.

        Calculates and stores the following regression metrics based on the predicted values:
        1. R-squared (coefficient of determination)
        2. Mean squared error
        3. Mean absolute error

        Parameters:
            None

        Returns:
            self: Modeler
                Returns the instance of the Modeler class.

        Example usage:
            modeler = Modeler()
            modeler.calculate_regression_metrics()
        """

        y_pred: pd.Series = self.regression_estimator.predict(self.X_test)

        r2 = r2_score(y_true=self.y_test, y_pred=y_pred)
        mean_squared_err = mean_squared_error(y_true=self.y_test, y_pred=y_pred)
        mean_absolute_err = mean_absolute_error(y_true=self.y_test, y_pred=y_pred)

        regression_metrics: RegressionMetrics = RegressionMetrics(
            r2=r2,
            mean_squared_error=mean_squared_err,
            mean_absolute_error=mean_absolute_err,
        )

        self.regression_metrics = regression_metrics

        return self

    def calculate_shap(
            self
    ) -> Modeler:

        """
        Calculate SHAP values and generate a summary plot.

        Parameters:
            self (Modeler): The modeler object.

        Returns:
            Modeler: The updated modeler object.

        Raises:
            ValueError: If neither a regression model nor a classifier model has been trained.

        Example:
            modeler = Modeler()
            modeler.calculate_shap()
        """

        if self.regression_estimator is not None:
            explainer = shap.Explainer(model=self.regression_estimator)
            shap_values = explainer(self.X_test)

        elif self.classifier_estimator is not None:
            explainer = shap.Explainer(model=self.classifier_estimator)
            shap_values = explainer(self.X_test)

        else:
            raise ValueError("Either Regression Model or Classifier must have been trained")

        shap.summary_plot(shap_values, self.X_test)

        return self

    def predict_this(
            self,
            x_df: pd.DataFrame
    ) -> pd.Series:

        """
        Parameters
        ----------
        x_df : pd.DataFrame
            The input data to be predicted.

        Returns
        -------
        y_pred : pd.Series
            The predicted values based on the input data.
        """

        x_columns: List[str] = x_df.columns.to_list()
        feature_columns: List[str] = self.X_train.columns.to_list()

        if x_columns != feature_columns:
            raise ValueError(f"x_columns {x_columns} does not match feature_columns {feature_columns}")

        y_pred: pd.Series = self.regression_estimator.predict(x_df)

        return y_pred

    def classify_this(
            self,
            x_df: pd.DataFrame
    ) -> pd.Series:
        """
        Parameters
        ----------
        x_df: pd.DataFrame
            The input DataFrame representing the feature values for classification.

        Returns
        -------
        y_pred: pd.Series
            The predicted class labels for the input features.

        Raises
        ------
        ValueError
            If the columns in the input DataFrame do not match the feature columns used for training the classifier.
        """
        x_columns: List[str] = x_df.columns.to_list()
        feature_columns: List[str] = self.X_train.columns.to_list()

        if x_columns != feature_columns:
            raise ValueError(f"x_columns {x_columns} does not match feature_columns {feature_columns}")

        y_pred: pd.Series = self.classifier_estimator.predict(x_df)

        return y_pred
