import pickle
import optuna

if __name__ == '__main__':
    with open("test_study.pkl", "rb") as f:
        study = pickle.load(f)

    print("Best trial:", study.best_trial.number)
    print("Best accuracy:", study.best_trial.value)
    print("Best hyperparameters:", study.best_params)

    fig = optuna.visualization.plot_param_importances(study)  # 파라미터 중요도 확인 그래프
    fig.show()
