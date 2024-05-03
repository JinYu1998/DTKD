import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


import numpy as np



class DTKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DTKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DTKD.CE_WEIGHT
        self.alpha = cfg.DTKD.ALPHA
        self.beta = cfg.DTKD.BETA
        self.warmup = cfg.DTKD.WARMUP
        self.temperature = cfg.DTKD.T

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # DTKD Loss
        reference_temp = self.temperature
        logits_student_max, _ = logits_student.max(dim=1, keepdim=True)
        logits_teacher_max, _ = logits_teacher.max(dim=1, keepdim=True)
        logits_student_temp = 2 * logits_student_max / (logits_teacher_max + logits_student_max) * reference_temp 
        logits_teacher_temp = 2 * logits_teacher_max / (logits_teacher_max + logits_student_max) * reference_temp
        
        ourskd = nn.KLDivLoss(reduction='none')(
            F.log_softmax(logits_student / logits_student_temp, dim=1) , # 学生
            F.softmax(logits_teacher / logits_teacher_temp, dim=1)       # 老师
        ) 
        loss_ourskd = (ourskd.sum(1, keepdim=True) * logits_teacher_temp * logits_student_temp).mean()
        
        # Vanilla KD Loss
        vanilla_temp = self.temperature
        kd = nn.KLDivLoss(reduction='none')(
            F.log_softmax(logits_student / vanilla_temp, dim=1) , # 学生
            F.softmax(logits_teacher / vanilla_temp, dim=1)       # 老师
        ) 
        loss_kd = (kd.sum(1, keepdim=True) * vanilla_temp ** 2).mean() 
         
        # CrossEntropy Loss
        loss_ce = nn.CrossEntropyLoss()(logits_student, target)

        loss_dtkd = min(kwargs["epoch"] / self.warmup, 1.0) * (self.alpha * loss_ourskd + self.beta * loss_kd) + self.ce_loss_weight * loss_ce
        losses_dict = {
            "loss_dtkd": loss_dtkd,
        }

        return logits_student, losses_dict
