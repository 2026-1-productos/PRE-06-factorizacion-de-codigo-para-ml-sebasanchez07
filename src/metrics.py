"""Funciones para calcular metricas de un modelo."""


def compute_metrics(y_true, y_pred):
    """Calcula MSE, MAE y R2 entre valores reales y predichos."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
    }


def print_metrics(metrics, title="Metricas"):
    """Imprime las metricas de forma bonita."""
    print()
    print(f"{title}:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
