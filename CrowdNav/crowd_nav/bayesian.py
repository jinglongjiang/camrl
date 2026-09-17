"""Public velocity-innovation beliefs and compositional local action values.

No private human types, goals, simulator lookahead or robot-response model.
"""
import math
import numpy as np
import torch
from torch import nn


SCHEMA = 'local-predictive-q-v3'
ARMS = ('current', 'prior', 'map', 'history', 'full')


def action_table(config):
    speeds = config.getint('policy', 'n_speeds')
    headings = config.getint('policy', 'n_headings')
    low = config.getfloat('policy', 'v_min')
    high = config.getfloat('policy', 'v_max')
    if not 0 < low <= high or speeds < 1 or headings < 1:
        raise ValueError('Invalid discrete action grid')
    u = np.linspace(0., 1., speeds)
    if config.get('policy', 'sampling', fallback='linear') == 'exponential':
        u = np.expm1(u) / np.expm1(1.)
    table = [[v*math.cos(a), v*math.sin(a)] for v in low+(high-low)*u
             for a in np.arange(headings)*2*math.pi/headings]
    if config.getboolean('policy', 'include_stop', fallback=False):
        table.insert(0, [0., 0.])
    return np.asarray(table, np.float32)


def batch_observations(observations, device='cpu'):
    if not observations:
        raise ValueError('Empty batch')
    maximum = max(1,max(len(o['humans']) for o in observations))
    result = {}
    for key in observations[0]:
        values = []
        for obs in observations:
            value = obs[key]
            if key in ('humans','modes','conditional','mask'):
                value = np.pad(value, [(0,maximum-len(value))]+[(0,0)]*(value.ndim-1))
            values.append(value)
        result[key] = torch.as_tensor(np.stack(values),dtype=torch.float32,device=device)
    return result


class MotionBelief:
    """Per-track Normal-Inverse-Gamma posterior for velocity innovations.

    This is an approximate stationary residual model, not a hidden human type.
    Public consecutive velocities are the only evidence. No robot response or
    private goals enter the filter. Five equal-weight points match its diagonal
    predictive covariance; they do not exactly integrate Student-t tails.
    """
    def __init__(self, config):
        if config.getboolean('robot', 'visible'):
            raise ValueError('Local motion protocol requires robot.visible=false')
        self.dt = config.getfloat('env', 'time_step')
        self.scale = config.getfloat('bayesian', 'innovation_std', fallback=.15)
        if self.scale <= 0:
            raise ValueError('innovation_std must be positive')
        self.reset()

    def reset(self):
        self.tracks = {}
        self.recent = {}
        self.frame = None

    def observe(self, raw, frame, remaining, arm='full', ids=None):
        if arm not in ARMS:
            raise ValueError('Unknown motion arm')
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 1 or len(raw) < 9 or (len(raw)-9)%5 or not np.isfinite(raw).all():
            raise ValueError('Invalid public observation')
        r, h = raw[:9], raw[9:].reshape(-1, 5)
        ids = tuple(range(len(h))) if ids is None else tuple(ids)
        if len(ids) != len(h) or len(set(ids)) != len(ids):
            raise ValueError('Unique persistent track IDs required')
        if self.frame is not None and frame not in (self.frame, self.frame+1):
            raise ValueError('Consecutive frames required; reset between episodes')
        means, variances = [], []
        for key, human in zip(ids, h):
            known = key in self.tracks
            if key not in self.tracks:
                self.tracks[key] = [human[2:4].copy(), 1., np.zeros(2), 2.,
                                    np.full(2, self.scale**2/2.)]
                self.recent[key] = []
            state = self.tracks[key]
            if known and self.frame is not None and frame != self.frame:
                last, k, mu, alpha, beta = state
                innovation = human[2:4]-last
                self.recent[key] = (self.recent[key]+[innovation.copy()])[-4:]
                delta = innovation-mu
                state = [human[2:4].copy(), k+1., mu+delta/(k+1.),
                         alpha+.5, beta+.5*k/(k+1.)*delta**2]
                self.tracks[key] = state
            _, k, mu, alpha, beta = state
            if arm == 'history' and self.recent[key]:
                window = np.asarray(self.recent[key])
                means.append(window.mean(0))
                # Frequentist plug-in prediction using four observed increments.
                variances.append(window.var(0,ddof=1)*(1.+1./len(window))
                                 if len(window)>1 else np.full(2,self.scale**2))
            else:
                means.append(mu.copy())
                variances.append(beta*(k+1.)/(k*(alpha-1.)))
        self.frame = frame
        mean = np.asarray(means).reshape(-1, 2)
        var = np.asarray(variances).reshape(-1, 2)
        if arm in ('current', 'prior'):
            mean[:] = 0.
            var[:] = 0. if arm == 'current' else self.scale**2
        elif arm == 'map':
            var[:] = 0.
        points = np.repeat((h[:, 2:4]+mean)[:, None, :], 5, axis=1)
        offset = np.sqrt(2.5*var)
        points[:, 1, 0] += offset[:, 0]
        points[:, 2, 0] -= offset[:, 0]
        points[:, 3, 1] += offset[:, 1]
        points[:, 4, 1] -= offset[:, 1]
        direction = r[2:4]-r[:2]
        angle = math.atan2(direction[1], direction[0])
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c,s],[-s,c]])
        physical = np.concatenate(((h[:,:2]-r[:2])@rotation.T/10.,
            (h[:,2:4]-r[4:6])@rotation.T/2., h[:,4:5]/.3), axis=1)
        ego = np.array([np.linalg.norm(direction)/10., *(rotation@r[4:6]/2.),
                        r[6]/.3,r[7]/2.,remaining,0.,0.], np.float32)
        return dict(robot=ego, humans=physical.astype(np.float32),
                    modes=((points-r[4:6])@rotation.T/2.).astype(np.float32),
                    conditional=np.full((len(h),5),.2,np.float32),
                    weights=np.full(5,.2,np.float32),
                    mask=np.ones(len(h),np.float32)), rotation


class LocalValueNetwork(nn.Module):
    """Shared local costs composed without attention dilution or fixed slots.

    Posterior integration occurs after the nonlinear local cost. A learned
    positive mixture of worst-person and total costs represents bottlenecks
    and accumulated exposure. This is an inductive bias, not a safety bound
    or a claim of exact value factorization.
    """
    def __init__(self, actions, width=128):
        super().__init__()
        self.register_buffer('actions', torch.as_tensor(actions,dtype=torch.float32))
        self.base = nn.Sequential(nn.Linear(8,width),nn.ReLU(),nn.Linear(width,1))
        self.local = nn.Sequential(nn.Linear(12,width),nn.ReLU(),
                                   nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
        self.mix = nn.Parameter(torch.zeros(2))

    def forward(self, obs):
        h, r = obs['humans'], obs['robot']
        mask = obs['mask'].bool()
        actions = self.actions[None].expand(len(r),-1,-1)
        ego = r[:,None,:6].expand(-1,len(self.actions),-1)
        base = self.base(torch.cat((ego,actions),-1)).squeeze(-1)
        # Mask before nonlinear arithmetic so padded values cannot leak.
        h = torch.where(mask[...,None],h,torch.zeros_like(h))
        modes = torch.where(mask[:,:,None,None],obs['modes'],torch.zeros_like(obs['modes']))
        position = h[:,None,:,None,:2]*10.
        relative_v = 2.*modes[:,None]+2.*r[:,None,None,None,1:3]-actions[:,:,None,None]
        radius = .3*(h[:,None,:,None,4]+r[:,None,None,None,3])
        distance = position.norm(dim=-1)
        closest_t = (-(position*relative_v).sum(-1)/relative_v.square().sum(-1).clamp_min(1e-6)).clamp(0.,2.)
        closest = (position+closest_t[...,None]*relative_v).norm(dim=-1)-radius
        shape = relative_v.shape[:-1]
        features = torch.cat((position.expand(*shape,2)/10., relative_v/2.,
            radius.expand(shape)[...,None],closest[...,None],closest_t[...,None]/2.,
            actions[:,:,None,None].expand(*shape,2),
            r[:,None,None,None,5].expand(shape)[...,None],
            h[:,None,:,None,2:4].expand(*shape,2)),-1)
        penalty = torch.nn.functional.softplus(self.local(features).squeeze(-1))
        # Local support is based only on current public geometry, same for all arms.
        relevance = ((6.-distance)/4.).clamp(0.,1.)
        cost = (penalty*obs['weights'][:,None,None]).sum(-1)*relevance[...,0]
        cost = cost*mask[:,None]
        mixture = self.mix.softmax(0)
        return base-mixture[0]*cost.max(-1).values-mixture[1]*cost.sum(-1)
