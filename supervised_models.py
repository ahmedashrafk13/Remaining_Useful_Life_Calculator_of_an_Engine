import pickle
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM,
                                     Bidirectional, Dense, Dropout,
                                     SpatialDropout1D, LayerNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# ── Mixed precision: ~2× GPU throughput on Tensor Core GPUs ──────────────────
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# ── NASA scoring ──────────────────────────────────────────────────────────────
def nasa_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


# ── constants ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 60
RUL_CLIP = 130
ROLL_WIN = 5
COLUMNS = (['engine_id', 'cycle'] +
           [f'op_setting_{i}' for i in range(1, 4)] +
           [f'sensor_{i}' for i in range(1, 22)])
LOW_VAR_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                   'sensor_16', 'sensor_18', 'sensor_19']


# ── vectorised rolling stats ───────────────────────────────────────────────────
def add_rolling(df: pd.DataFrame, cols: list, win: int) -> pd.DataFrame:
    df = df.copy()
    grped = df.groupby('engine_id')[cols]
    roll = grped.rolling(win, min_periods=1)

    means = roll.mean().reset_index(level=0, drop=True)
    stds = roll.std().fillna(0).reset_index(level=0, drop=True)

    means.columns = [f'{c}_mean{win}' for c in cols]
    stds.columns = [f'{c}_std{win}' for c in cols]

    df = pd.concat([df, means, stds], axis=1)
    return df


# ── vectorised cluster-wise normalisation ─────────────────────────────────────
def apply_cluster_norm(df: pd.DataFrame,
                       stats: pd.DataFrame,
                       feat_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in feat_cols:
        col_mean = df['op_cluster'].map(stats[(col, 'mean')])
        col_std = df['op_cluster'].map(stats[(col, 'std')])
        # only normalise where std is meaningful (avoids divide-by-zero)
        mask = col_std > 1e-6
        df.loc[mask, col] = (df.loc[mask, col] - col_mean[mask]) / col_std[mask]
    return df


# ── preprocessing ─────────────────────────────────────────────────────────────
def preprocess(train_path, test_path, rul_path):
    """
    Pipeline order (must match app.py exactly):
        1. Read → assign RUL labels
        2. MinMaxScaler on raw sensor + op_setting cols (fit on train only)
        3. Drop low-variance sensors
        4. Add rolling mean/std  (vectorised)
        5. Cluster-wise z-score normalisation (condition compensation, vectorised)
    FIX: scaler is returned AND saved to scaler.pkl so app.py can load it.
    """
    train_df = pd.read_csv(train_path, sep=" ", header=None).dropna(axis=1)
    test_df = pd.read_csv(test_path, sep=" ", header=None).dropna(axis=1)
    rul_df = pd.read_csv(rul_path, header=None)

    train_df.columns = COLUMNS
    test_df.columns = COLUMNS
    rul_df.columns = ['RUL']
    rul_df['engine_id'] = rul_df.index + 1

    # ── RUL labels ────────────────────────────────────────────────────────────
    max_cyc = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cyc.columns = ['engine_id', 'max_cycle']
    train_df = pd.merge(train_df, max_cyc, on='engine_id')
    train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=RUL_CLIP)
    train_df.drop('max_cycle', axis=1, inplace=True)

    test_last = test_df.groupby('engine_id')['cycle'].max().reset_index()
    test_last.columns = ['engine_id', 'last_cycle']
    test_rul = pd.merge(test_last, rul_df, on='engine_id')
    test_df = pd.merge(test_df, test_rul, on='engine_id')
    test_df['RUL'] = (test_df['RUL'] +
                      (test_df['last_cycle'] - test_df['cycle'])).clip(upper=RUL_CLIP)
    test_df.drop('last_cycle', axis=1, inplace=True)

    # ── step 2: MinMaxScaler ──────────────────────────────────────────────────
    scaler = MinMaxScaler()
    features = train_df.columns.difference(['engine_id', 'cycle', 'RUL'])
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # save scaler so app.py can load the identical fitted scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  Scaler saved → scaler.pkl")

    # ── step 3: drop low-variance sensors ─────────────────────────────────────
    to_drop = [s for s in LOW_VAR_SENSORS if s in train_df.columns]
    train_df = train_df.drop(columns=to_drop)
    test_df = test_df.drop(columns=to_drop)

    sensor_cols = [c for c in train_df.columns
                   if c not in ['engine_id', 'cycle', 'RUL']]

    # ── step 4: rolling stats (vectorised) ────────────────────────────────────
    train_df = add_rolling(train_df, sensor_cols, ROLL_WIN)
    test_df = add_rolling(test_df, sensor_cols, ROLL_WIN)

    # ── step 5: condition-wise normalisation (vectorised) ─────────────────────
    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
    train_df['op_cluster'] = kmeans.fit_predict(train_df[op_cols])
    test_df['op_cluster'] = kmeans.predict(test_df[op_cols])

    sensor_feat_cols = [c for c in train_df.columns
                        if c not in ['engine_id', 'cycle', 'RUL', 'op_cluster']]
    cluster_stats = train_df.groupby('op_cluster')[sensor_feat_cols].agg(['mean', 'std'])

    train_df = apply_cluster_norm(train_df, cluster_stats, sensor_feat_cols)
    test_df = apply_cluster_norm(test_df, cluster_stats, sensor_feat_cols)

    train_df = train_df.drop(columns=['op_cluster'])
    test_df = test_df.drop(columns=['op_cluster'])

    return train_df, test_df, scaler


# ── sequence builder (pre-allocated, off-by-one fixed) ────────────────────────
def create_sequences(df: pd.DataFrame,
                     feature_cols: list,
                     seq_len: int):
    # first pass: count total windows
    total = 0
    for _, group in df.groupby('engine_id'):
        n = len(group)
        if n > seq_len:
            total += n - seq_len + 1  # FIX: was n - seq_len (off by one)

    n_feat = len(feature_cols)
    X_seq = np.empty((total, seq_len, n_feat), dtype=np.float32)
    y_seq = np.empty(total, dtype=np.float32)

    idx = 0
    for _, group in df.groupby('engine_id'):
        group = group.sort_values('cycle')
        features = group[feature_cols].values.astype(np.float32)
        targets = group['RUL'].values.astype(np.float32)
        n = len(features)
        for i in range(n - seq_len + 1):  # FIX: inclusive upper bound
            X_seq[idx] = features[i: i + seq_len]
            y_seq[idx] = targets[i + seq_len - 1]
            idx += 1

    return X_seq[:idx], y_seq[:idx]


# ── attention layer ───────────────────────────────────────────────────────────
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1, activation='tanh', dtype='float32')
        self.softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, x):
        # cast to float32 for numerical stability when using mixed precision
        x = tf.cast(x, tf.float32)
        scores = self.score_dense(x)
        weights = self.softmax(scores)
        weights = tf.cast(weights, dtype=x.dtype)
        return tf.reduce_sum(x * weights, axis=1)

    def get_config(self):
        return super().get_config()


# ── model builder ─────────────────────────────────────────────────────────────
def build_cnn_lstm_attention(seq_len: int, n_features: int) -> Model:
    reg = regularizers.l2(1e-4)

    inp = Input(shape=(seq_len, n_features))

    # CNN block
    x = Conv1D(64, 3, activation='relu', padding='same',
               kernel_regularizer=reg)(inp)
    x = Conv1D(64, 3, activation='relu', padding='same',
               kernel_regularizer=reg)(x)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.3)(x)

    # Bi-LSTM block — NO recurrent_dropout (keeps CuDNN kernel active)
    x = Bidirectional(
        LSTM(64, return_sequences=True,
             dropout=0.2)  # FIX: recurrent_dropout removed
    )(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(
        LSTM(32, return_sequences=True,
             dropout=0.2)  # FIX: recurrent_dropout removed
    )(x)
    x = Dropout(0.2)(x)

    # Attention (outputs float32)
    x = TemporalAttention()(x)

    # Dense head
    x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = LayerNormalization(dtype='float32')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', kernel_regularizer=reg)(x)
    # output must be float32 even under mixed precision
    out = Dense(1, dtype='float32')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(3e-4, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=10.0),
        metrics=['mae']
    )
    return model


# ── tf.data pipeline ──────────────────────────────────────────────────────────
def make_dataset(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """
    FIX (GPU perf): wraps numpy arrays in a tf.data pipeline with
    cache + optional shuffle + batch + prefetch(AUTOTUNE).
    prefetch keeps the GPU fed while the CPU prepares the next batch.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  LOADING & PREPROCESSING DATA")
    print("=" * 60)
    train_df, test_df, scaler = preprocess(
        'train_FD002.txt', 'test_FD002.txt', 'RUL_FD002.txt'
    )
    print(f"Train shape: {train_df.shape}  |  Test shape: {test_df.shape}")

    feature_cols = [c for c in train_df.columns
                    if c not in ['engine_id', 'cycle', 'RUL']]

    # ── Engine-based split for CNN-LSTM (prevents sequence leakage) ───────────
    all_engine_ids = sorted(train_df['engine_id'].unique())
    rng = np.random.default_rng(42)
    shuffled_ids = rng.permutation(all_engine_ids)
    n_val = int(len(shuffled_ids) * 0.2)
    val_engine_ids = shuffled_ids[:n_val].tolist()
    tr_engine_ids = shuffled_ids[n_val:].tolist()

    tr_df = train_df[train_df['engine_id'].isin(tr_engine_ids)]
    val_df = train_df[train_df['engine_id'].isin(val_engine_ids)]
    print(f"Train engines: {len(tr_engine_ids)}  |  Val engines: {len(val_engine_ids)}")

    # ── Row-level split for sklearn models ────────────────────────────────────
    X_all = train_df[feature_cols]
    y_all = train_df['RUL']
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    all_results = {}

    # ── Linear Regression ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING: Linear Regression")
    print("=" * 60)
    lr = LinearRegression()
    lr.fit(X_train_sk, y_train_sk)
    y_p = lr.predict(X_test_sk)
    rmse = float(np.sqrt(mean_squared_error(y_test_sk, y_p)))
    mae = float(mean_absolute_error(y_test_sk, y_p))
    print(f"  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
    all_results['Linear Regression'] = {
        'rmse': rmse, 'mae': mae,
        'nasa_score': nasa_score(y_test_sk.values, y_p),
        'y_pred': y_p.tolist(), 'y_true': y_test_sk.tolist(),
    }

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING: Random Forest")
    print("=" * 60)
    rf = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1,
        min_samples_leaf=4, max_features=0.7
    )
    rf.fit(X_train_sk, y_train_sk)
    y_p = rf.predict(X_test_sk)
    rmse = float(np.sqrt(mean_squared_error(y_test_sk, y_p)))
    mae = float(mean_absolute_error(y_test_sk, y_p))
    print(f"  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
    all_results['Random Forest'] = {
        'rmse': rmse, 'mae': mae,
        'nasa_score': nasa_score(y_test_sk.values, y_p),
        'y_pred': y_p.tolist(), 'y_true': y_test_sk.tolist(),
    }

    # ── CNN-LSTM + Attention ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING: CNN-LSTM + Attention")
    print("=" * 60)

    X_tr, y_tr = create_sequences(tr_df, feature_cols, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_df, feature_cols, SEQUENCE_LENGTH)
    print(f"  Train sequences: {X_tr.shape}  |  Val sequences: {X_val.shape}")

    # FIX (GPU perf): tf.data pipelines with prefetch
    BATCH_SIZE = 64
    train_ds = make_dataset(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = build_cnn_lstm_attention(SEQUENCE_LENGTH, len(feature_cols))
    model.summary()

    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        epochs=100,
        validation_data=val_ds,
        callbacks=[
            checkpoint,
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
        ],
        verbose=1
    )

    # ── Validation metrics ────────────────────────────────────────────────────
    y_pred_val = model.predict(val_ds).flatten()
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    val_mae = float(mean_absolute_error(y_val, y_pred_val))
    val_nasa = nasa_score(y_val, y_pred_val)
    print(f"\n  VAL  — RMSE: {val_rmse:.4f}  |  MAE: {val_mae:.4f}  |  NASA: {val_nasa:.2f}")

    # ── True test set evaluation ──────────────────────────────────────────────
    print("\n  Evaluating on true test set (test_FD002.txt)...")


    def build_test_sequences(df, feature_cols, seq_len):
        """
        Last-cycle prediction per engine (NASA convention).
        FIX (bug): padding now repeats the LAST row (most-degraded state),
        not the first row (healthy state). A nearly-failed engine should be
        padded with its end-of-life readings, not its initial healthy ones.
        """
        X_seq, y_seq = [], []
        for _, group in df.groupby('engine_id'):
            group = group.sort_values('cycle')
            vals = group[feature_cols].values.astype(np.float32)
            target = group['RUL'].values[-1]
            if len(vals) >= seq_len:
                seq = vals[-seq_len:]
            else:

                pad = np.repeat(vals[-1:], seq_len - len(vals), axis=0)
                seq = np.vstack([pad, vals])
            X_seq.append(seq)
            y_seq.append(target)
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


    X_test_seq, y_test_seq = build_test_sequences(test_df, feature_cols, SEQUENCE_LENGTH)
    test_ds = make_dataset(X_test_seq, y_test_seq, BATCH_SIZE, shuffle=False)
    y_pred_test = np.clip(model.predict(test_ds).flatten(), 0, RUL_CLIP)
    test_rmse = float(np.sqrt(mean_squared_error(y_test_seq, y_pred_test)))
    test_mae = float(mean_absolute_error(y_test_seq, y_pred_test))
    test_nasa = nasa_score(y_test_seq, y_pred_test)
    print(f"  TEST — RMSE: {test_rmse:.4f}  |  MAE: {test_mae:.4f}  |  NASA: {test_nasa:.2f}")

    all_results['CNN-LSTM + Attention'] = {
        'rmse': val_rmse,
        'mae': val_mae,
        'nasa_score': val_nasa,
        'y_pred': y_pred_val.tolist(),
        'y_true': y_val.tolist(),
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_nasa_score': test_nasa,
        'test_y_pred': y_pred_test.tolist(),
        'test_y_true': y_test_seq.tolist(),
        'history_loss': history.history['loss'],
        'history_val_loss': history.history['val_loss'],
        'history_mae': history.history['mae'],
        'history_val_mae': history.history['val_mae'],
    }

    # ── save results ──────────────────────────────────────────────────────────
    with open('cnn_lstm_results.json', 'w') as f:
        json.dump(all_results, f)

    print("\n" + "=" * 60)
    print("  ALL RESULTS SAVED → cnn_lstm_results.json")
    print("  Scaler saved       → scaler.pkl")
    print("  Now open Streamlit: streamlit run app.py")
    print("=" * 60)

    print("\n  MODEL COMPARISON (Validation Scores):")
    print(f"  {'Model':<25} {'RMSE':>8}  {'MAE':>8}  {'NASA':>10}")
    print("  " + "-" * 55)
    for name, v in all_results.items():
        ns = v.get('nasa_score', float('nan'))
        print(f"  {name:<25} {v['rmse']:>8.4f}  {v['mae']:>8.4f}  {ns:>10.2f}")

    print("\n  CNN-LSTM TRUE TEST SET SCORES:")
    cnn = all_results['CNN-LSTM + Attention']
    print(f"  RMSE:       {cnn['test_rmse']:.4f}")
    print(f"  MAE:        {cnn['test_mae']:.4f}")
    print(f"  NASA Score: {cnn['test_nasa_score']:.2f}")
