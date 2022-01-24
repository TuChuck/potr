###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Implments the model function for the POTR model."""


from ast import arg
import numpy as np
import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.H36MDataset_v2 as H36MDataset_v2
import data.NTURGDDataset as NTURGDDataset
import utils.utils as utils

import pandas as pd
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_WEIGHT_DECAY = 0.00001
_NSEEDS = 8

class POTRModelFn(seq2seq_model_fn.ModelFn):
  def __init__(self,
               params,
               train_dataset_fn,
               eval_dataset_fn,
               pose_encoder_fn=None,
               pose_decoder_fn=None):
    super(POTRModelFn, self).__init__(
      params, train_dataset_fn, eval_dataset_fn, pose_encoder_fn, pose_decoder_fn)
    self._loss_fn = self.layerwise_loss_fn

  def smooth_l1(self, decoder_pred, decoder_gt):
    l1loss = nn.SmoothL1Loss(reduction='mean')
    return l1loss(decoder_pred, decoder_gt)

  def loss_l1(self, decoder_pred, decoder_gt):
    return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

  def loss_activity(self, logits, class_gt):                                     
    """Computes entropy loss from logits between predictions and class."""
    return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

  def compute_class_loss(self, class_logits, class_gt):
    """Computes the class loss for each of the decoder layers predictions or memory."""
    class_loss = 0.0
    for l in range(len(class_logits)):
      class_loss += self.loss_activity(class_logits[l], class_gt)

    return class_loss/len(class_logits)

  def select_loss_fn(self):
    if self._params['loss_fn'] == 'mse':
      return self.loss_mse
    elif self._params['loss_fn'] == 'smoothl1':
      return self.smooth_l1
    elif self._params['loss_fn'] == 'l1':
      return self.loss_l1
    else:
      raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

  def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
    """Computes layerwise loss between predictions and ground truth."""
    pose_loss = 0.0
    loss_fn = self.select_loss_fn()

    for l in range(len(decoder_pred)):
      pose_loss += loss_fn(decoder_pred[l], decoder_gt)

    pose_loss = pose_loss/len(decoder_pred)
    if class_logits is not None:
      return pose_loss, self.compute_class_loss(class_logits, class_gt)

    return pose_loss, None

  def init_model(self, pose_encoder_fn=None, pose_decoder_fn=None):
    self._model = PoseTransformer.model_factory(
        self._params, 
        pose_encoder_fn, 
        pose_decoder_fn
    )

  def select_optimizer(self):
    optimizer = optim.AdamW(
        self._model.parameters(), lr=self._params['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=_WEIGHT_DECAY
    )

    return optimizer

  def save_euler_error_csv(self):
    self.finetune_init()

    _time = time.time()
    self._model.eval()
    self._scalars = {}
    
    head = 'action'
    head = np.append(head,[self._ms_range])

    eval_loss = self.evaluate_fn(0, _time)

    ms_eval_loss = self._scalars['ms_eval_loss']

    ms_mean = 'mean'
    ms_mean = np.append(ms_mean, [np.vstack(list(ms_eval_loss.values())).mean(axis=0)])
    for idx,k in enumerate(ms_eval_loss.keys()):
      _log = k
      df = pd.DataFrame(np.expand_dims(np.append(_log,[ms_eval_loss[k]]),axis=0))
      if idx == 0:
        df.to_csv(thispath+'/'+args['config_path'].split('/')[-1]+'.csv',header=head, index=False)
      else:
        with open(thispath+'/'+args['config_path'].split('/')[-1]+'.csv','a') as f:
          df.to_csv(f,header=False,index=False)
    
    ## add mean values across ms
    df = pd.DataFrame(np.expand_dims(ms_mean,axis=0))
    with open(thispath+'/'+args['config_path'].split('/')[-1]+'.csv','a') as f:
      df.to_csv(f,header=False,index=False)

def dataset_factory(params):
  if params['dataset'] == 'h36m_v2':
    return H36MDataset_v2.dataset_factory(params)
  elif params['dataset'] == 'ntu_rgbd':
    return NTURGDDataset.dataset_factory(params)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config_path", type=str, default="")
  
  args = vars(parser.parse_args())

  config_path = os.path.join(args['config_path'],'config','config.json')

  with open(config_path,'r') as _file:
    params = json.load(_file)

  ckpt_path = os.path.join(args['config_path'],'models','ckpt_epoch_0499.pt')
  params['finetuning_ckpt'] = ckpt_path

  train_dataset_fn, eval_dataset_fn = dataset_factory(params)

  params['input_dim'] = train_dataset_fn.dataset._data_dim
  params['pose_dim'] = train_dataset_fn.dataset._pose_dim
  pose_encoder_fn, pose_decoder_fn = \
      PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

  model_fn = POTRModelFn(
      params, train_dataset_fn, 
      eval_dataset_fn, 
      pose_encoder_fn, pose_decoder_fn
  )

  model_fn.save_euler_error_csv()