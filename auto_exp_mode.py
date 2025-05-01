import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from collections import deque
from matplotlib.widgets import Slider, CheckButtons
import tf2onnx
import onnx
import onnxruntime as ort
import ipywidgets as widgets
from IPython.display import display, clear_output

import streamlit as st
import plotly.express as px
import pandas as pd


from tools.lib.logreader import LogReader
from opendbc.can.parser import CANParser
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.basedir import BASEDIR

plt.ion()

# Corolla w/ new tune real drive
# lr1 = list(LogReader('a2bddce0b6747e10/000002ac--c04552b8af', sort_by_time=True))
# lr2 = list(LogReader('a2bddce0b6747e10/000002ab--4c8e1b86d9', sort_by_time=True))
# lr3 = list(LogReader('a2bddce0b6747e10/000002aa--9b130338b9', sort_by_time=True))
# lr4 = list(LogReader('a2bddce0b6747e10/000002ba--969fe70a70', sort_by_time=True))
# lr5 = list(LogReader('a2bddce0b6747e10/000002bb--3752fc5bba', sort_by_time=True))
# lr6 = list(LogReader('a2bddce0b6747e10/000002c8--f1df5a0b52', sort_by_time=True))
# lr7 = list(LogReader('a2bddce0b6747e10/000002c9--19a235a8d4', sort_by_time=True))

lrs = [
  (True, LogReader('d9b97c1d3b8c39b2/000000b6--4c41d698c4/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001f4--d15b86861c/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001d2--5052ad43c8/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001e5--eceae68fa2/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000c1--7712828b37/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000b6--4c41d698c4/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000a6--79e372b105/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/0000008f--d66e410a2a/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/0000008d--68d9a569e1/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001ec--6b5c4edefe/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000c1--7712828b37/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000c1--7712828b37/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000bc--1d3e454aa1/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000bb--ce04fddbee/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000b8--da4a4290aa/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000b7--6d46bd2283/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000c3--45657c2376/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000c2--ead5b24827/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000cf--4170a0b8c1/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000d0--371918fedc/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000d1--8209deb086/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/000000cf--4170a0b8c1/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/0000008f--d66e410a2a/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/0000008e--00a787d7fc/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/0000008d--68d9a569e1/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/00000082--544ad55f1a/q', sort_by_time=True)),
  (True, LogReader('d9b97c1d3b8c39b2/00000050--74e83a7056/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001fd--8785cf106f/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001fe--bb5a9536cd/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001fb--df219d064d/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/000001e8--7a35eff0d1/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/00000195--2826236027/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/00000187--d866d5b534/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/0000017f--f0cdc90388/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/0000015a--bb0978aced/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/00000159--e44e9326dc/q', sort_by_time=True)),
  (True, LogReader('2c912ca5de3b1ee9/00000158--cb7389803f/q', sort_by_time=True)),
  # (True, LogReader('NONENOENEONEONE', sort_by_time=True)),
  # (True, LogReader('NONENOENEONEONE', sort_by_time=True)),
]

# Corolla w/ new tune maneuvers
# lr8 = list(LogReader('a2bddce0b6747e10/000002a9--a207f8b605', sort_by_time=True))

DECIMATION = 2
SECTION_LEN = 60 * DECIMATION
STRIDE = 10 * DECIMATION

X_sections = []
Y_sections = []

X_speeds = deque(maxlen=SECTION_LEN)
X_accels = deque(maxlen=SECTION_LEN)
X_gas_pressed = deque(maxlen=SECTION_LEN)
X_lead_speeds = deque(maxlen=SECTION_LEN)
X_lead_dists = deque(maxlen=SECTION_LEN)
X_lead_accels = deque(maxlen=SECTION_LEN)
X_model_curvatures = deque(maxlen=SECTION_LEN)
X_model_accelerations = deque(maxlen=SECTION_LEN)

tracked_bufs = [
  X_speeds, X_accels, X_gas_pressed,
  X_lead_speeds, X_lead_dists, X_lead_accels, X_model_curvatures, X_model_accelerations,
]


def reset_data():
  for buf in tracked_bufs:
    buf.clear()

# Y_experimental_modes = []

for stock_route, lr in tqdm(lrs):
  if not lr.first('carParams').openpilotLongitudinalControl:
    continue
  # cp = CANParser("toyota_nodsu_pt_generated", [
  #   ("PCM_CRUISE", 33),
  #   ("CLUTCH", 15),
  #   ("VSC1S07", 20),
  # ], 0)
  # cp2 = CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 33)], 2)
  # cp128 = CANParser("toyota_nodsu_pt_generated", [("ACC_CONTROL", 33)], 128)
  reset_data()

  CC = None
  CS = None
  CP = None
  LP = None
  CO = None
  RD = None
  MDL = None
  prev_new_accel = 0
  long_active_frames = 0
  frame = 0

  for msg in tqdm(lr):
    if msg.which() == 'carControl':
      CC = msg.carControl

    elif msg.which() == 'radarState':
      RD = msg.radarState

    elif msg.which() == 'carParams':
      CP = msg.carParams

    elif msg.which() == 'livePose':
      LP = msg.livePose

    elif msg.which() == 'carState':
      CS = msg.carState

    elif msg.which() == 'drivingModelData':
      MDL = msg.drivingModelData

    elif msg.which() == 'selfdriveState':
      if not RD or not CS or not CC or not MDL:
        continue

      frame += 1

      if not CC.enabled and CS.standstill:
        reset_data()
        continue

      if frame % 5 != 0:  # gather at 2hz
        continue

      X_speeds.append(CS.vEgo)
      X_accels.append(CS.aEgo)
      X_gas_pressed.append(CS.gasPressed)
      X_lead_speeds.append(RD.leadOne.vLeadK)
      X_lead_dists.append(RD.leadOne.dRel)
      X_lead_accels.append(RD.leadOne.aLeadK)
      X_model_curvatures.append(MDL.action.desiredCurvature)
      X_model_accelerations.append(MDL.action.desiredAcceleration)

      # print(set(map(len, tracked_bufs)))
      if set(map(len, tracked_bufs)) == {SECTION_LEN}:
        X_section = list(zip(*tracked_bufs, strict=True))
        # print(X_section)
        X_sections.append(X_section)
        Y_sections.append(msg.selfdriveState.experimentalMode)
        # print(Y_sections[-1])
        for buf in tracked_bufs:
          for _ in range(STRIDE):
            buf.popleft()

# raise Exception

# # keep track of sections because data is not continuous
# # delay cmd (y) by 15 frames so that it roughly matches the result (x)
# X, Y = [], []
# for x_section, y_section in zip(X_sections, Y_sections):
#   # trim off first 0.5s after engaging
#   X.extend(x_section[15:][50:])
#   Y.extend(y_section[:-15][50:])

X = np.array(X_sections)
Y = np.array(Y_sections)
assert len(X) == len(Y)
print('Samples', len(X))

# def plot_data_stats():
#   # scatter plot where x is aEgo and y is accel cmd
#   fig, ax = plt.subplots(1)
#   ax.scatter([x[1] for x in X], [y[0] for y in Y], s=1)
#   ax.plot([-5, 5], [-5, 5], 'r--', label='y=x')
#   ax.set_xlabel('aEgo')
#   ax.set_ylabel('accel cmd')
#   ax.set_title('aEgo vs accel cmd')
#   ax.set_xlim(-5, 5)
#   ax.set_ylim(-5, 5)
#   plt.legend()
#   plt.show()
#   # plt.pause(100)
#
#
# plot_data_stats()


# # FIXME: this one is vibe coded and terrible, but cool idea
# def plot_data_stats2():
#   # Assumes X and Y are defined in the global scope.
#   vEgo = np.array([x[0] for x in X])
#   aEgo = np.array([x[1] for x in X])
#   pitch = np.array([x[2] for x in X])
#   accel_cmd = np.array([y[0] for y in Y])
#
#   # --- Plot setup ---
#   fig, ax = plt.subplots()
#   # Adjust layout to accommodate sliders and toggles.
#   plt.subplots_adjust(bottom=0.3, right=0.8)
#
#   # Plot the initial scatter using ALL the points (before filtering).
#   sc = ax.scatter(aEgo, accel_cmd, c=pitch, s=3, cmap='viridis')
#   ax.set_xlabel('aEgo')
#   ax.set_ylabel('accel_cmd')
#   ax.set_title('aEgo vs accel_cmd colored by pitch')
#   cb = plt.colorbar(sc, ax=ax)
#   cb.set_label("Pitch")
#
#   # --- Add y = x reference line ---
#   # Create a reference line over the range of aEgo values
#   x_line = np.linspace(np.min(aEgo), np.max(aEgo), 100)
#   ax.plot(x_line, x_line, 'r--', label='y = x')
#   ax.legend(loc='upper left')
#
#   # --- Slider axes for filter centers ---
#   ax_speed = plt.axes([0.15, 0.2, 0.65, 0.03])
#   ax_pitch = plt.axes([0.15, 0.15, 0.65, 0.03])
#   # Slider for vEgo center with its range set to that of vEgo.
#   s_speed = Slider(ax_speed, 'vEgo Center', vEgo.min(), vEgo.max(), valinit=vEgo.mean())
#   # Slider for pitch center (in rad) with its range set to that of pitch.
#   s_pitch = Slider(ax_pitch, 'Pitch Center (rad)', pitch.min(), pitch.max(), valinit=pitch.mean())
#
#   # --- Toggle checkboxes ---
#   # Place checkboxes in an axes on the right.
#   ax_check = plt.axes([0.82, 0.4, 0.15, 0.15])
#   # Two toggles: one to apply the speed filter and one for the pitch filter.
#   check = CheckButtons(ax_check, ['Speed Filter', 'Pitch Filter'], [True, True])
#
#   def update(val):
#     speed_center = s_speed.val
#     pitch_center = s_pitch.val
#
#     # If the speed filter is enabled, select points within ±5 m/s.
#     if check.get_status()[0]:
#       mask_speed = np.abs(vEgo - speed_center) <= 5.0
#     else:
#       mask_speed = np.ones_like(vEgo, dtype=bool)
#     # If the pitch filter is enabled, select points within ±5° (in radians).
#     if check.get_status()[1]:
#       mask_pitch = np.abs(pitch - pitch_center) <= np.deg2rad(5)
#     else:
#       mask_pitch = np.ones_like(pitch, dtype=bool)
#
#     mask = mask_speed & mask_pitch
#
#     # Decimate the data if too many points are selected
#     N_selected = np.sum(mask)
#     decimation_threshold = 10000  # adjust threshold as needed
#     if N_selected > decimation_threshold:
#       indices = np.where(mask)[0]
#       chosen_indices = np.random.choice(indices, decimation_threshold, replace=False)
#       current_mask = np.zeros_like(mask, dtype=bool)
#       current_mask[chosen_indices] = True
#     else:
#       current_mask = mask
#
#     # Update scatter plot with filtered (and possibly decimated) data.
#     offsets = np.column_stack((aEgo[current_mask], accel_cmd[current_mask]))
#     sc.set_offsets(offsets)
#     sc.set_array(pitch[current_mask])
#     fig.canvas.draw_idle()
#
#   s_speed.on_changed(update)
#   s_pitch.on_changed(update)
#   check.on_clicked(lambda label: update(None))
#
#   update(None)  # initial draw
#   plt.show()
#
#
# # Call the function to run the visualization.
# plot_data_stats2()

# raise Exception
# train model to simulate aEgo from requested accel

# the model

normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(X)

USE_CUDNN = False
# DROPOUT = 0.5

inputs = keras.layers.Input(shape=X.shape[1:])
shared = normalizer(inputs)  # scale
# shared = keras.layers.GRU(16, use_cudnn=USE_CUDNN, return_sequences=True, recurrent_dropout=DROPOUT, dropout=DROPOUT)(shared)
shared = keras.layers.GRU(32, use_cudnn=USE_CUDNN)(shared)
shared = keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001))(shared)
# shared = keras.layers.BatchNormalization()(shared)
shared = keras.layers.LeakyReLU()(shared)
# shared = keras.layers.Dropout(DROPOUT)(shared)

# shared = keras.layers.Dense(8, kernel_regularizer=keras.regularizers.l2(0.01))(shared)
# shared = keras.layers.LeakyReLU()(shared)
# shared = keras.layers.Dropout(DROPOUT)(shared)

exp_output = keras.layers.Dense(1, activation='sigmoid', name='exp_output')(shared)

model = keras.models.Model(inputs=inputs, outputs=exp_output)

model.compile(
  optimizer='adam', loss={'exp_output': 'binary_crossentropy'},
  metrics={'exp_output': ['accuracy']},
)

model.summary()

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

class_weight = {True: np.mean(~Y_train), False: np.mean(Y_train)}

callbacks = [
  # keras.callbacks.EarlyStopping(monitor='val_loss',
  #                               patience=3,
  #                               restore_best_weights=True),
  # keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
  #                                   factor=0.5,
  #                                   patience=2)
]

try:
  model.fit(X_train, Y_train, batch_size=64, epochs=40, shuffle=True,
            validation_data=(X_test, Y_test), class_weight=class_weight,
            callbacks=callbacks)
except KeyboardInterrupt:
  pass


def plot_model_prediction():
  fig, ax = plt.subplots(1)
  speeds = np.linspace(5, 35, 4)
  for speed in speeds:
    x = np.linspace(-4, 4, 10)
    y = model.predict(np.stack([np.full_like(x, speed), x, np.zeros_like(x), np.ones_like(x)], axis=1))
    ax.plot(x, y[0].flatten(), 'o-', label=f'speed {speed:.1f} m/s')
    ax.set_xlabel('aEgo')
    ax.set_ylabel('accel cmd')
    ax.set_title('aEgo vs accel cmd')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
  plt.show()


# plot_model_prediction()

# raise Exception

# save model
# model.output_names=['output']
print("Saving model")
spec = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
onnx.save(onnx_model, BASEDIR + '/exp_mode.onnx')

X_pred = model.predict(X)
predicted_camera_accel = X_pred[0].flatten()
predicted_permit_braking = X_pred[1].flatten()

loaded_model = ort.InferenceSession(BASEDIR + '/exp_mode.onnx')

predicted_camera_accel_loaded = loaded_model.run(None, {'input': X.astype(np.float32)})[0].flatten()

# plot prediction
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot([y[0] for y in Y], label='accel cmd (output)')
ax[0].plot(predicted_camera_accel, label='predicted accel cmd (output)')
ax[0].plot([x[1] for x in X], label='aEgo (input)')

# ax[0].plot(X_accels, label='actuatorsOutput.accel')
# ax[0].plot(y_aegos, label='aEgo (ground)')
# # ax[0].plot(y_aegos, label='accelerationDevice.x (ground)')
# ax[0].plot(predicted_aegos, label='predicted aEgo')
# # ax[0].plot(predicted_aegos_nn, label='predicted aEgo (NN)')
# # ax[0].plot(predicted_aegos_nn2, label='predicted aEgo (branch NN)')
# # ax[0].plot(X_accels_past, label='past accel')
ax[0].legend()

# permit braking (output) and mini car (input)
ax[1].plot([y[1] for y in Y], label='permit braking (output)')
ax[1].plot(predicted_permit_braking, label='predicted permit braking (output)')
ax[1].plot([x[3] for x in X], label='mini car (input)')
ax[1].legend()

ax[2].plot([x[0] for x in X], label='vEgo')
ax[2].legend()

plt.show()
