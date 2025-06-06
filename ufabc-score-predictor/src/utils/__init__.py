def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)