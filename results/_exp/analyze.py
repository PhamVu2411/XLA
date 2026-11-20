import sys, time, numpy as np, cv2
sys.path.insert(0,'python'); sys.path.insert(0,'tests')
import image_ops as ops
from sift_extractor import SIFT
from panorama_stitching import Panorama
from run_panorama import read_rgb, save_rgb
from compare_with_opencv import CASES
s=SIFT(num_downsamples=0)
for name,fname,loader in CASES:
    ims=loader(); p=Panorama()
    prep=p._prepare(ims)
    print('==',name)
    for i,im in enumerate(prep):
        g=ops.rgb2gray(im); gx,gy=ops.sobel(g); mag=np.sqrt(gx**2+gy**2)
        kp=len(s.extract_sift_data(g))
        print(f'  anh{i+1} {im.shape[1]}x{im.shape[0]} sang={g.mean():.2f} tuongphan={g.std():.2f} grad={mag.mean():.3f} kp={kp}')
    # reprojection error of inliers
    gray=[ops.rgb2gray(x) for x in prep]; sd=p.get_sift_descriptors(gray)
    for i in range(1,len(prep)):
        a,b=sd[i-1],sd[i]
        da=np.stack(a.Descriptor.to_list()); db=np.stack(b.Descriptor.to_list())
        pa=np.array(a.Position.to_list(),float); pb=np.array(b.Position.to_list(),float)
        i1,i2,_=ops.match_descriptors(da,db,1.0,0.8)
        H,m=ops.ransac_homography(pa[i1],pb[i2],5.0)
        e=np.linalg.norm(ops.apply_homography(H,pa[i1][m])-pb[i2][m],axis=1)
        print(f'  cap {i}-{i+1}: khop={len(i1)} inlier={m.sum()} ({m.mean()*100:.0f}%) loi_chieu_TB={e.mean():.2f}px dx={H[0,2]:.0f}')
