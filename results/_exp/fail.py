import sys, time, numpy as np
sys.path.insert(0,'python')
import image_ops as ops
from panorama_stitching import Panorama
from run_panorama import read_rgb, save_rgb
I=lambda n: read_rgb('images/'+n)
m1,m2=I('mosque1.jpg'),I('mosque2.jpg'); img=I('img.jpg')
def dark(x,f): return (np.clip(x/255.*f,0,1)*255).astype(np.uint8)
def rot(x,a): return (ops.rotate(x/255.,a)*255).astype(np.uint8)
sky=img[:300]  # vùng trời ít chi tiết?
cases={
 'F1_sai_thu_tu':[m2,m1],
 'F2_khong_chong_lan':[m1, I('img2.jpg')],
 'F3_anh_toi':[m1, dark(m2,0.35)],
 'F4_xoay_30':[m1, rot(m2,30)],
 'F5_chong_lan_it':[img[:, :700], img[:, 640:1366]],
 'F6_nhieu':[m1, np.clip(m2+np.random.default_rng(0).normal(0,40,m2.shape),0,255).astype(np.uint8)],
}
for k,ims in cases.items():
    p=Panorama(); t=time.time()
    try:
        r=p.get_panorama(ims); save_rgb(f'results/_exp/{k}.jpg',r)
        print(k,'OK', r.shape[1],'x',r.shape[0], p.match_info, f'{time.time()-t:.1f}s')
    except Exception as e:
        print(k,'LOI:',e, p.match_info)
