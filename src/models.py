from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def get_models(random_state: int = 42) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.0001, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "SVR": SVR(C=10, epsilon=0.1, kernel='rbf'),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "VotingRegressor": VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=200, random_state=random_state)),
                ('svr', SVR(C=10, epsilon=0.1, kernel='rbf'))
            ],
            weights=[0.3, 0.7], n_jobs=-1
        )
    }
