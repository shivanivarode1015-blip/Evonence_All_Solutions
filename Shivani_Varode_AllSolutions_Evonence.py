#1)Scenario 1: Data ValidationTask
def validate_data(data):
    def check(entry):
        try:
            assert isinstance(entry,dict), "Not a dict"
            assert "age" in entry, "Missing age"
            assert isinstance(entry["age"], int), "Age not integer"
            assert entry["age"] >= 0, "Age negative"
        except AssertionError as e:
            return {"entry": entry, "error": str(e)}
        return None
    return list(filter(None,(check(item) for item in data)))

#2)Scenario 2: Logging DecoratorTask
import time
from functools import wraps
from contextlib import contextmanager
@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[LOG] {name} executed in {end - start: .6f} seconds")
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper
@log_execution_time
def calculate_sum(n):
    return n * (n + 1) // 2
result = calculate_sum(1000000)
print("Sum: " , result)

#3)Scenario 3: Missing Value Handling
import pandas as pd
def smart_fill_income(df):
    skew_val = df["income"].skew()
    strategy = {
        True: df["income"].mode()[0],     
        False: df["income"].median()      
    }
    fill_value = strategy[abs(skew_val) > 0.5]
    return df.assign(
        income=lambda x: x["income"].fillna(fill_value)
    ), skew_val, fill_value
data = {
    "income": [50000, 60000, None, 55000, None, 700000]
}
df = pd.DataFrame(data)
updated_df, skewness, used_value = smart_fill_income(df)

print("Skewness:", skewness)
print("Value used for filling:", used_value)
print(updated_df)

#4)Scenario 4: Text Pre-processing
import pandas as pd
import re
def preprocess_text(df, col):
    pipeline = [
        str.lower,
        lambda x: re.sub(r'[^a-z0-9\s]', '', x),  
        lambda x: (token for token in x.split() if token) 
    ]
    def apply_pipeline(text):
        for step in pipeline:
            text = step(text)
        return list(text)  
    df[col] = df[col].map(apply_pipeline)
    return df
data = {
    "text": ["Hello!!! World@", "AI & ML is #1", "Python@@@ Programming!!!"]
}
df = pd.DataFrame(data)
cleaned_df = preprocess_text(df, "text")
print(cleaned_df)


#5)Scenario 5: Hyperparameter Tuning
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
def smart_grid_search(X, y):
    def custom_score(y_true, y_pred, **kwargs):
        score = f1_score(y_true, y_pred, average='weighted')
        depth_penalty = kwargs.get("estimator").max_depth * 0.001
        return score - depth_penalty

    scorer = make_scorer(custom_score)
    param_grid = {
        "max_depth": list({d for d in [3, 5, 7]}),  # set → list (uncommon style)
        "n_estimators": list(np.linspace(50, 100, 2, dtype=int))
    }

    model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        return_train_score=True
    )

    grid.fit(X, y)

    # Extract top 3 configurations (rare insight)
    results = grid.cv_results_
    ranked = sorted(
        zip(results['mean_test_score'], results['params']),
        reverse=True
    )[:3]

    return grid.best_params_, ranked


# Example usage (assuming X, y already defined)
best_params, top3 = smart_grid_search(X, y)

print("Best Parameters:", best_params)
print("Top 3 Configurations:")
for score, params in top3:
    print(score, params)


#6)Scenario 6: Custom Evaluation Metric
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer
def weighted_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
 
    weights = np.array([1, 2])
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    class_accuracy = np.divide(correct, total, where=total!=0)
    weighted_acc = np.dot(class_accuracy, weights) / weights.sum() 
    return weighted_acc
weighted_scorer = make_scorer(weighted_accuracy)
y_true = np.array([0, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])
print("Weighted Accuracy:", weighted_accuracy(y_true, y_pred))


#7)Scenario 7: Image Augmentation
import tensorflow as tf
from tensorflow.keras import layers
class RandomApply(layers.Layer):
    def __init__(self, layer, prob=0.5):
        super().__init__()
        self.layer = layer
        self.prob = prob
    def call(self, x, training=True):
        if training:
            return tf.cond(
                tf.random.uniform(()) < self.prob,
                lambda: self.layer(x),
                lambda: x
            )
        return x
def build_augmentation():
    return tf.keras.Sequential([
        RandomApply(layers.RandomRotation(0.055), prob=0.7),  
        RandomApply(layers.RandomFlip("horizontal"), prob=0.5),
        RandomApply(layers.RandomZoom(0.2), prob=0.6),
    ], name="smart_augmentation")
def augment_dataset(dataset):
    aug_model = build_augmentation()
    return dataset.map(
        lambda x, y: (aug_model(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


#8)Scenario 8: Model Callbacks
import tensorflow as tf
class SmartEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
            print(f"Improved validation loss at epoch {epoch+1}: {current_loss:.4f}")
        else:
            self.wait += 1
            print(f"No improvement for {self.wait} epoch(s)")
        if self.wait >= self.patience:
            print("Stopping early...")
            self.model.stop_training = True
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                print("Best weights restored")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stopping = SmartEarlyStopping(patience=3)


#9)Scenario 9: Structured Response Generation
import google.generativeai as genai
import json
import re
genai.configure(api_key="YOUR_API_KEY")
def extract_json(text):
    """
    Try to extract JSON even if response is messy
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group()
    return None
def safe_json_load(text):
    """
    Self-healing JSON loader
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = text.replace("'", '"')  
        fixed = re.sub(r",\s*}", "}", fixed)  
        try:
            return json.loads(fixed)
        except:
            return {"error": "Invalid JSON", "raw_output": text}
def get_structured_response():
    model = genai.GenerativeModel("gemini-pro")
    prompt = """
    Give response ONLY in JSON format like:
    {
        "benefits": ["...", "...", "..."]
    }
    Query: List 3 benefits of Python for data science.
    """
    response = model.generate_content(prompt)
    raw_text = response.text
    json_text = extract_json(raw_text)
    if json_text:
        return safe_json_load(json_text)
    else:
        return {"error": "No JSON found", "raw_output": raw_text}
result = get_structured_response()
print(result)


#10)Scenario 10: Summarization with Constraints
import re
def summarize(article, model):
    res = model.generate_content(f"Summarize in exactly 2 sentences under 50 words:\n{article}")
    s = res.text.strip()
    if len(s.split()) > 50:
        s = " ".join(re.split(r'(?<=[.!?]) +', s)[:2])
    return s
