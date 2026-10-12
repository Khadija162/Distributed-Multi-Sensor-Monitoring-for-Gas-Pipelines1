"""From-paper TeDKF research implementation, not authors' official code.

The paper experiments use a 4-state thermal system, NOT a gas-flow model.
Each sensor operates an individual 4D local KF. Each step fuses ONLY innovation
vectors using a trusted set selected against preceding trusted innovations.
"""
from dataclasses import dataclass
import numpy as np

A=np.array([[.8,.1,0,0],[.1,.8,.1,0],[0,.1,.8,.1],[0,0,.1,.8]],dtype=float)

def mutual_reachability(reference, candidate, gamma=5):
    """Eq. (8): max(core(candidate), core(reference), pair distance).

    Reference points are previous trusted innovations, with core distance
    from their gamma-th closest OTHER reference. This is a documented
    finite-sample implementation choice.
    """
    ref=np.asarray(reference,dtype=float)
    point=np.asarray(candidate,dtype=float)
    if ref.ndim!=2 or ref.shape[0]<2 or point.shape!=(ref.shape[1],):
        raise ValueError("Need at least two reference innovations of matching dimension")
    count=min(int(gamma),len(ref)-1)
    differences=np.linalg.norm(ref-point,axis=1)
    candidate_core=np.partition(differences,min(count-1,len(ref)-1))[count-1]
    pair=np.linalg.norm(ref[:,None,:]-ref[None,:,:],axis=2)
    np.fill_diagonal(pair,np.inf)
    core_ref=np.partition(pair,count-1,axis=1)[:,count-1]
    return np.maximum.reduce([np.full(len(ref),candidate_core),core_ref,differences])

def trusted_innovations(innovations,reference,lam=.35,gamma=5):
    """IDB inspired by Eqs. (6)-(9): metrics, adaptive threshold, trusted set."""
    values=np.asarray(innovations,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError("Invalid innovations")
    if reference is None:
        return np.ones(len(values),dtype=bool),np.zeros(len(values)),float("inf")
    scores=np.array([mutual_reachability(reference,v,gamma).min() for v in values])
    threshold=float(scores.mean()+lam)  # Paper Eq. (9)
    trusted=scores<=threshold
    # Never silently label the lowest score as trusted if every point fails.
    return trusted,scores,threshold

@dataclass
class Step:
    state: list
    trusted: list
    detection_scores: list
    threshold: float|None
    innovation: list

class TeDKF:
    """Paper-aligned local KF, innovation fusion, and one-step trust memory."""
    def __init__(self,n_sensors=10,initial=(20.,22.,24.,23.),
                 q=.05,r=.1,lam=.35,gamma=5):
        if n_sensors<3 or q<=0 or r<=0 or gamma<1 or lam<0:
            raise ValueError("Invalid filter hyperparameters")
        self.n=n_sensors
        self.x=np.tile(np.asarray(initial,dtype=float),(n_sensors,1))
        self.p=np.tile(np.eye(4)[None,:,:],(n_sensors,1,1))
        self.q=q*np.eye(4)
        self.r=r*np.eye(4)
        self.lam=lam
        self.gamma=gamma
        self.reference=None
        self.t=0

    def step(self,observations):
        z=np.asarray(observations,dtype=float)
        if z.shape!=(self.n,4) or not np.isfinite(z).all():
            raise ValueError("Observations must be finite, shape=(sensors,4)")
        prior=self.x@A.T
        cov=np.array([A@p@A.T+self.q for p in self.p])
        innovations=z-prior  # H = I, Eq. (5)
        trusted,scores,tau=trusted_innovations(innovations,self.reference,self.lam,self.gamma)
        # The first nominal snapshot seeds trust as required by Algorithm 2.
        if trusted.any():
            fused=np.mean(innovations[trusted],axis=0)  # Eq. (10)
            self.reference=innovations[trusted].copy()
        else:
            # Fail closed: prediction-only update, never fuse known-untrusted nodes.
            fused=np.zeros(4)
            # Retain prior trusted reference for recovery.
        for i in range(self.n):
            gain=cov[i]@np.linalg.inv(cov[i]+self.r)
            self.x[i]=prior[i]+gain@fused
            # Joseph covariance update prevents numerical asymmetry.
            identity=np.eye(4)
            self.p[i]=(identity-gain)@cov[i]@(identity-gain).T+gain@self.r@gain.T
        self.t+=1
        return Step(state=self.x.mean(axis=0).tolist(),
                    trusted=trusted.tolist(),
                    detection_scores=scores.tolist(),
                    threshold=None if not np.isfinite(tau) else tau,
                    innovation=fused.tolist())
