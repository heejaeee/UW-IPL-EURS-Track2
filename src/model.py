from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2
from torchvision.models.segmentation.lraspp import LRASPPHead
def resize(x,size):return F.interpolate(x,size=size,mode='bilinear',align_corners=False)

class ColorBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c,c*2,1),nn.ReLU(),nn.Conv2d(c*2,c*2,3,padding=1,groups=c*2),nn.ReLU(),nn.Conv2d(c*2,c,1))
    def forward(self,x):return x+self.net(x)

class TinyRestorer(nn.Module):
    """Raw-image-only spatial color coefficients, with no shared semantic weights."""
    def __init__(self):
        super().__init__()
        self.e1=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),ColorBlock(24))
        self.e2=nn.Sequential(nn.Conv2d(24,32,3,stride=2,padding=1),nn.ReLU(),ColorBlock(32))
        self.e3=nn.Sequential(nn.Conv2d(32,48,3,stride=2,padding=1),nn.ReLU(),ColorBlock(48),ColorBlock(48))
        self.d2=nn.Sequential(nn.Conv2d(80,32,1),nn.ReLU(),ColorBlock(32))
        self.d1=nn.Sequential(nn.Conv2d(56,24,1),nn.ReLU(),ColorBlock(24))
        self.local=nn.Conv2d(24,6,3,padding=1)
        self.global_color=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(48,32,1),nn.ReLU(),nn.Conv2d(32,6,1))
        nn.init.zeros_(self.local.weight);nn.init.zeros_(self.local.bias)
        nn.init.zeros_(self.global_color[-1].weight);nn.init.zeros_(self.global_color[-1].bias)
    def forward(self,image):
        small=F.interpolate(image,size=(192,256),mode='bilinear',align_corners=False)
        a=self.e1(small);b=self.e2(a);c=self.e3(b)
        up=lambda v,skip:F.interpolate(v,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        z=self.d2(torch.cat((up(c,b),b),1));z=self.d1(torch.cat((up(z,a),a),1))
        raw=up(self.local(z),image)+self.global_color(c)
        gain=(1+raw[:,:3]).clamp(.05,12)-1
        shift=raw[:,3:6]
        # Keep a scalar zero in ONNX instead of materializing a196608-element
        # constant initializer that the official counter counts as parameters.
        zero=gain[:,0:1]*0.
        return torch.cat((gain[:,0:1],zero,zero,zero,gain[:,1:2],zero,zero,zero,gain[:,2:3],shift),1)

class GridRestorer(TinyRestorer):
    def __init__(self, size=(192,256), detail=False):
        super().__init__(); self.grid=tuple(size)
        if detail: self.detail_gain=nn.Parameter(torch.zeros(1))

    def forward(self, image):
        small=resize(image,self.grid)
        a=self.e1(small); z=self.e2(a); c=self.e3(z)
        up=lambda x,y:resize(x,y.shape[-2:])
        z=self.d2(torch.cat((up(c,z),z),1));z=self.d1(torch.cat((up(z,a),a),1))
        raw=up(self.local(z),image)+self.global_color(c)
        gain=(1+raw[:,:3]).clamp(.05,12)-1;shift=raw[:,3:6];zero=gain[:,:1]*0.
        return torch.cat((gain[:,:1],zero,zero,zero,gain[:,1:2],zero,zero,zero,gain[:,2:3],shift),1)

def color(diagonal,shift,off=None):
    if off is None:off=diagonal[:,:1].expand(-1,6,-1,-1)*0
    return torch.cat((diagonal[:,:1],off[:,:2],off[:,2:3],diagonal[:,1:2],off[:,3:5],off[:,5:6],diagonal[:,2:3],shift),1)

class CrossRestorer(GridRestorer):
    def __init__(self,kernel=1):
        super().__init__((192,256))
        self.cross_local=nn.Conv2d(24,6,kernel,padding=kernel//2)
        self.cross_global=nn.Conv2d(32,6,1)
        for m in (self.cross_local,self.cross_global):nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)

    def forward(self,image):
        x=resize(image,self.grid);a=self.e1(x);b=self.e2(a);c=self.e3(b)
        up=lambda x,y:resize(x,y.shape[-2:])
        z=self.d2(torch.cat((up(c,b),b),1));z=self.d1(torch.cat((up(z,a),a),1))
        h=self.global_color[2](self.global_color[1](self.global_color[0](c)))
        raw=up(self.local(z),image)+self.global_color[3](h)
        off=up(self.cross_local(z),image)+self.cross_global(h)
        return color((1+raw[:,:3]).clamp(.05,12)-1,raw[:,3:],off)

def apply_color(image,color):
    residual=color[:,:3]*image[:,0:1]+color[:,3:6]*image[:,1:2]+color[:,6:9]*image[:,2:3]+color[:,9:12]
    return (image+residual).clamp(0,1)

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=mobilenet_v2(weights=None).features[:15]
        self.features[7].conv[1][0].stride=(1,1)
        for i in range(8,14):
            self.features[i].conv[1][0].dilation=(2,2)
            self.features[i].conv[1][0].padding=(2,2)
        self.features[14].conv[1][0].stride=(1,1)
        self.features[14].conv[1][0].dilation=(4,4)
        self.features[14].conv[1][0].padding=(4,4)
        self.head=LRASPPHead(24,160,5,128)
        self.restorer=CrossRestorer()
        self.blend_logit=nn.Parameter(torch.zeros(1,3,1,1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.register_buffer('logit_bias',torch.zeros(1,5,1,1))
        self.guided=False;self.detail=False
        self.budget_config=dict(whole=(384,512),tile=(192,288),context_weight=.75)
    def mix(self,x,restored,detach=False):
        alpha=.35*self.blend_logit.sigmoid()
        mixed=x+alpha*((restored.detach() if detach else restored)-x)
        return x+.25*(mixed-x)
    def raw_features(self,image):
        x=(image-self.mean)/self.std
        for i,layer in enumerate(self.features):
            x=layer(x)
            if i==3:low=x
        high=x;z=self.head.cbr(high)*self.head.scale(high)
        logits=self.head.low_classifier(low)+self.head.high_classifier(resize(z,low.shape[-2:]))
        return resize(logits,image.shape[-2:])+self.logit_bias,z,high

class TiledStudent(nn.Module):
    def __init__(self,model,config):
        super().__init__();self.model=model;self.config=config
        # Shared 128-element constants; never materialize a 1024x1024 mask.
        self.register_buffer('ramp',(torch.arange(128).float()+.5)/128)

    def semantic(self,image,size):
        small=F.interpolate(image,size=size,mode='bilinear',align_corners=False)
        return self.model.semantic_features(small)[0]

    def merge(self,a,b,axis):
        if axis==-1:
            w=self.ramp.view(1,1,1,128)
            return torch.cat((a[:,:,:,:448],a[:,:,:,448:]* (1-w)+b[:,:,:,:128]*w,b[:,:,:,128:]),-1)
        w=self.ramp.view(1,1,128,1)
        return torch.cat((a[:,:,:448,:],a[:,:,448:,:]*(1-w)+b[:,:,:128,:]*w,b[:,:,128:,:]),-2)

    def forward(self,image):
        small=F.interpolate(image,size=(384,512),mode='bilinear',align_corners=False)
        color=self.model.restorer(small)
        color=F.interpolate(color,size=(1024,1024),mode='bilinear',align_corners=False)
        restored=apply_color(image,color)
        whole=None
        if self.config['whole'] is not None:
            whole=F.interpolate(self.semantic(image,self.config['whole']),size=(1024,1024),mode='bilinear',align_corners=False)
        if self.config['tile'] is None:return restored,whole
        tiles=[]
        for top,left in ((0,0),(0,448),(448,0),(448,448)):
            patch=image[:,:,top:top+576,left:left+576]
            value=self.semantic(patch,self.config['tile'])
            tiles.append(F.interpolate(value,size=(576,576),mode='bilinear',align_corners=False))
        top=self.merge(tiles[0],tiles[1],-1);bottom=self.merge(tiles[2],tiles[3],-1)
        local=self.merge(top,bottom,-2)
        if whole is not None:
            weight=self.config['context_weight'];local=local*(1-weight)+whole*weight
        return restored,local

class ColorStudent(Student):
    def __init__(self,use_norm):
        super().__init__();self.use_norm=use_norm
        self.register_buffer('color_mean',torch.zeros(1,3,1,1))
        self.register_buffer('color_std',torch.ones(1,3,1,1))
        self.color_beta=nn.Parameter(torch.zeros(1,3,1,1))
    def mix(self,x,restored,detach=False):
        mixed=super().mix(x,restored,detach)
        if not self.use_norm:return mixed
        mu=mixed.mean((2,3),keepdim=True)
        std=((mixed-mu).square().mean((2,3),keepdim=True)+1e-4).sqrt()
        normal=(mixed-mu)/std*self.color_std+self.color_mean
        return mixed+self.color_beta.tanh()*(normal-mixed)

class SeparableUSM(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.exp(-torch.arange(-2, 3, dtype=torch.float32).square()/2)
        g = g/g.sum()
        self.register_buffer('luma', torch.tensor([.299, .587, .114]).view(1,3,1,1))
        self.register_buffer('horizontal', g.view(1,1,1,5))
        self.register_buffer('vertical', g.view(1,1,5,1))

    def forward(self, rgb):
        y = F.conv2d(rgb, self.luma)
        blurred = F.conv2d(F.pad(y, (2,2,2,2), mode='reflect'), self.horizontal)
        blurred = F.conv2d(blurred, self.vertical)
        return (rgb + .8*(y-blurred)).clamp(0,1)
from restoration.statistical_restorer_v3s import StatisticalRestorerV3S
from restoration.statistical_restorer_v3 import condition_grid

class Merged(nn.Module):
    def __init__(self,root,use_norm):
        super().__init__()
        student=ColorStudent(use_norm)
        student.restorer=StatisticalRestorerV3S(grid=condition_grid(Path(root)/'weights/all_train_fits.jsonl'))
        self.native=TiledStudent(student,dict(whole=(384,512),tile=(192,280),context_weight=.75))
        self.usm=SeparableUSM()
    def forward(self,image):
        rgb,whole,tiles=branches(self,image)
        whole,local=reconstruct(whole,tiles,self.native.merge)
        return rgb,refine_search(whole,local,rgb,SEARCH_CONFIG)

def load_bundle(root,device='cpu'):
    import json
    root=Path(root);config=json.loads((root/'config.json').read_text())
    model=Merged(root,config['use_norm'])
    model.load_state_dict(torch.load(root/config['weights'],map_location='cpu',weights_only=True)['model'],strict=True)
    return model.to(device).eval()


def refine(logits,image,restored,config):
 out=logits
 if config.get('radius',0):
  guide=image if config['guide']=='raw' else restored
  guide=.299*guide[:,0:1]+.587*guide[:,1:2]+.114*guide[:,2:3]
  r=config['radius'];eps=config['epsilon']
  box=lambda x:F.avg_pool2d(x,2*r+1,1,r,count_include_pad=False)
  mu=box(guide);p=box(logits)
  var=(box(guide*guide)-mu*mu).clamp_min(0)
  cov=box(guide*logits)-mu*p
  coef=cov/(var+eps);offset=p-coef*mu
  filtered=box(coef)*guide+box(offset)
  out=logits+config['strength']*(filtered-logits)
 if config.get('opening',0):
  k=config['opening'];values=[]
  for c in range(5):
   z=out[:,c:c+1]
   if c in (3,4):
    other=torch.cat((out[:,:c],out[:,c+1:]),1).amax(1,keepdim=True)
    margin=z-other
    eroded=-F.max_pool2d(-margin,k,1,k//2)
    opened=F.max_pool2d(eroded,k,1,k//2)
    z=z-(margin-opened).clamp(min=0,max=config['opening_cap'])
   values.append(z)
  out=torch.cat(values,1)
 return out

def box(x,r):
    # Separable box average; truncated border normalization on each axis.
    x=F.avg_pool2d(x,(1,2*r+1),1,(0,r),count_include_pad=False)
    return F.avg_pool2d(x,(2*r+1,1),1,(r,0),count_include_pad=False)

def guided(z,g,r=8,eps=.001,rgb=False):
    if not rgb:
        g=.299*g[:,0:1]+.587*g[:,1:2]+.114*g[:,2:3]
        m=box(g,r); q=box(z,r)
        a=(box(g*z,r)-m*q)/((box(g*g,r)-m*m).clamp_min(0)+eps)
        return box(a,r)*g+box(q-a*m,r)
    m=box(g,r);q=box(z,r)
    channels=[g[:,i:i+1] for i in range(3)]
    means=[m[:,i:i+1] for i in range(3)]
    cov={}
    for i in range(3):
        for j in range(i,3):
            v=box(channels[i]*channels[j],r)-means[i]*means[j]
            cov[i,j]=v.clamp_min(0)+eps if i==j else v
    a,b,c,d,e,f=[cov[k] for k in ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))]
    cross=[box(channels[i]*z,r)-means[i]*q for i in range(3)]
    # Apply the exact 3x3 inverse through LDL^T elimination. A determinant
    # near eps^3=1e-9 underflows in FP16; these positive pivots stay >=eps.
    # Schur complements of a PSD covariance + eps*I are >=eps in exact math.
    l10=b/a;l20=c/a
    pivot1=(d-b*l10).clamp_min(eps)
    l21=(e-c*l10)/pivot1
    pivot2=(f-c*l20-l21*l21*pivot1).clamp_min(eps)
    rhs1=cross[1]-l10*cross[0]
    rhs2=cross[2]-l20*cross[0]-l21*rhs1
    a2=rhs2/pivot2
    a1=rhs1/pivot1-l21*a2
    a0=cross[0]/a-l10*a1-l20*a2
    coeff=[a0,a1,a2]
    offset=q-sum(coeff[i]*means[i] for i in range(3))
    return sum(box(coeff[i],r)*channels[i] for i in range(3))+box(offset,r)

def branches(base,x):
    model=base.native.model;rgb=base.usm(model.restorer(x));mixed=model.mix(x,x)
    whole=model.raw_features(resize(mixed,(384,512)))[0]
    tiles=[model.raw_features(resize(mixed[:,:,t:t+576,l:l+576],(192,280)))[0] for t,l in POSITIONS]
    return rgb,whole,tiles

def reconstruct(whole,tiles,merger):
    parts=[resize(z,(576,576)) for z in tiles]
    local=merger(merger(parts[0],parts[1],-1),merger(parts[2],parts[3],-1),-2)
    return resize(whole,(1024,1024)),local

def refine_search(whole,local,rgb,cfg):
    if cfg['family']=='baseline':
        return refine(.75*whole+.25*local,rgb,rgb,dict(guide='restored',radius=8,epsilon=.001,strength=1.))
    w=cfg.get('weight',.75);r=cfg.get('radius',8);eps=cfg.get('epsilon',.001)
    color=cfg.get('color',False);g=rgb
    filt=lambda z,rr=r:guided(z,g,rr,eps,color)
    family=cfg['family']
    if family=='probability':
        # Softmax AFTER each branch is reconstructed at deployment resolution.
        return w*filt(whole.softmax(1))+(1-w)*filt(local.softmax(1))
    z=w*whole+(1-w)*local
    if family=='class_radius':
        return torch.cat([guided(z[:,i:i+1],g,rr,eps,color) for i,rr in enumerate(cfg['radii'])],1)
    if family=='two_stage':return guided(filt(z,16),g,4,eps,color)
    return filt(z)

POSITIONS=((0, 0), (0, 448), (448, 0), (448, 448))
SEARCH_CONFIG={'family': 'two_stage', 'color': True}

class TTA(nn.Module):
    def __init__(self, root):
        super().__init__()
        self.base = Merged(root, True)
    def forward(self, x):
        m = self.base.native.model
        rgb = self.base.usm(m.restorer(x))
        mixed = m.mix(x, x)
        whole_x = resize(mixed, [384, 512])
        whole = m.raw_features(whole_x)[0]
        whole = (whole + m.raw_features(whole_x.flip(-1))[0].flip(-1)) * .5
        z = resize(whole, (1024, 1024))
        return rgb, guided(guided(z, rgb, 16, .001, True), rgb, 4, .001, True)
