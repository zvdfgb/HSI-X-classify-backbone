"""Temperature KL in the teacher-to-student direction, without new parameters."""
import math
import torch
from torch.nn import functional as F


def temperature_kl(student, teacher, temperature=4.0, threshold=None):
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError('Teacher and student must share B x classes logits')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Confidence threshold must be in [0, 1]')
    teacher = teacher.detach()
    per_sample = F.kl_div(F.log_softmax(student / temperature, dim=1),
                          F.softmax(teacher / temperature, dim=1), reduction='none').sum(1) * temperature**2
    accepted = torch.ones(student.shape[0],dtype=torch.bool,device=student.device)
    if threshold is not None:
        accepted = teacher.softmax(1).amax(1) >= threshold
    loss=(per_sample * accepted).sum() / accepted.sum().clamp_min(1)
    return loss, accepted
